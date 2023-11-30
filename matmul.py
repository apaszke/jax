import pathlib
import jaxlib
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith
from jaxlib.mlir.dialects import func
from jaxlib.mlir.dialects import gpu
from jaxlib.mlir.dialects import memref
from jaxlib.mlir.dialects import nvgpu
from jaxlib.mlir.dialects import nvvm
from jaxlib.mlir.dialects import scf
from jaxlib.mlir.dialects import vector
from jaxlib.mlir.execution_engine import ExecutionEngine
from jaxlib.mlir.passmanager import PassManager
print(memref.__file__)

DYNAMIC = -9223372036854775808

def strides(xs):
  strides = []
  stride = 1
  for x in xs[::-1]:
    strides.append(stride)
    stride *= x
  return strides[::-1]


with ir.Context() as ctx, ir.Location.unknown():
  barrier_group_ty = ir.Type.parse(
      "!nvgpu.mbarrier.group<memorySpace = #gpu.address_space<workgroup>, num_barriers = 2>"
  )
  acc_ty = ir.Type.parse(
      "!nvgpu.warpgroup.accumulator<fragmented = vector<128x128xf32>>"
  )
  token_ty = ir.Type.parse("!gpu.async.token")
  lhs_tile_shape = lhs_tma_shape = (128, 64)
  lhs_tensor_map_ty = ir.Type.parse(
      "!nvgpu.tensormap.descriptor<tensor = memref<128x64xf16, 3>, swizzle = swizzle_128b, l2promo=none, oob=zero, interleave=none>"
  )
  rhs_tile_shape = (64, 128)
  rhs_tma_shape = (64, 64)
  rhs_tensor_map_ty = ir.Type.parse(
      "!nvgpu.tensormap.descriptor<tensor = memref<64x64xf16, 3>, swizzle = swizzle_128b, l2promo=none, oob=zero, interleave=none>"
  )

  f16 = ir.F16Type.get()
  f32 = ir.F32Type.get()
  index = ir.IndexType.get()
  i32 = ir.IntegerType.get_signless(32)
  i64 = ir.IntegerType.get_signless(64)
  smem = ir.IntegerAttr.get(i64, 3)

  m = ir.Module.create()
  with ir.InsertionPoint(m.body):
    empty_smem_f16_ty = ir.TypeAttr.get(ir.MemRefType.get((0,), f16, memory_space=smem))
    dynamic_smem_global = memref.GlobalOp("dynamicSmem", empty_smem_f16_ty, sym_visibility="private", alignment=16)
    acc_smem_global = memref.GlobalOp("accSmem", ir.TypeAttr.get(ir.MemRefType.get((0,), f32, memory_space=smem)), sym_visibility="private", alignment=16)

    @func.FuncOp.from_py_func()
    def main():
      # matrix A [128][64] * matrix B[64][128] * stages(2)
      smem_size = 65536
      def c(value, ty=index):
        return arith.ConstantOp(ty, value)

      a_ty = b_ty = ir.MemRefType.get((128, 128), f16)
      c_ty = ir.MemRefType.get((128, 128), f32)
      a_host = memref.AllocOp(a_ty, [], [])
      b_host = memref.AllocOp(b_ty, [], [])
      c_host = memref.AllocOp(c_ty, [], [])

      token = gpu.WaitOp(token_ty, [])
      a_device, _ = gpu.AllocOp(a_ty, token_ty, [token], [], []).results
      b_device, _ = gpu.AllocOp(b_ty, token_ty, [token], [], []).results
      c_device, _ = gpu.AllocOp(c_ty, token_ty, [token], [], []).results
      tma_specs = [
          (a_device, a_host, lhs_tensor_map_ty, lhs_tma_shape),
          (b_device, b_host, rhs_tensor_map_ty, rhs_tma_shape)
      ]
      tma_descs = []
      for x_device, x_host, tensor_map_ty, tile_shape in tma_specs:
        gpu.MemcpyOp(token_ty, [token], x_device, x_host)
        x_unranked = memref.CastOp(ir.UnrankedMemRefType.get(f16, a_ty.memory_space), x_device)
        tma_descs.append(nvgpu.TmaCreateDescriptorOp(tensor_map_ty, x_unranked, map(c, tile_shape)).result)
      a_tma_desc, b_tma_desc = tma_descs

      grid = (1, 1, 1)
      block = (128, 1, 1)
      launch_op = gpu.LaunchOp(
          token_ty, [token], *map(c, grid), *map(c, block),
          dynamicSharedMemorySize=c(smem_size, ty=i32))
      launch_op.body.blocks.append(*([index] * 12))  # Append an empty block
      with ir.InsertionPoint(launch_op.body.blocks[0]):
        memref.AssumeAlignmentOp(c_device, 16)
        dynamic_smem = memref.GetGlobalOp(dynamic_smem_global.type_.value, "dynamicSmem")
        a_smem_strides = strides([2, *lhs_tile_shape])
        a_smem = memref.ReinterpretCastOp(
            ir.MemRefType.get((2, *lhs_tile_shape), f16, memory_space=smem),
            dynamic_smem, [], [], [],
            [0], [2, *lhs_tile_shape], a_smem_strides)
        b_smem_strides = strides([4, *rhs_tile_shape])
        b_smem = memref.ReinterpretCastOp(
            ir.MemRefType.get((4, *rhs_tile_shape), f16, memory_space=smem),
            dynamic_smem, [], [], [],
            [0], [4, *rhs_tile_shape], b_smem_strides)
        b_smem = memref.SubViewOp(
            ir.Type.parse("memref<2x64x128xf16, strided<[8192, 128, 1], offset: 16384>, 3>"),
            b_smem, [], [], [],
            [2, 0, 0], [2, *rhs_tile_shape], [1, 1, 1])

        tidx = gpu.ThreadIdOp(gpu.Dimension.x)
        is_leader = arith.CmpIOp(arith.CmpIPredicate.eq, tidx, c(0))

        barrier_group = nvgpu.MBarrierCreateOp(barrier_group_ty)
        for i in range(2):
          nvgpu.MBarrierInitOp(barrier_group, c(1), c(i))
        for desc in tma_descs:
          nvgpu.TmaPrefetchOp(desc)

        def fetch(step: int | ir.Value):
          if isinstance(step, int):
            step = c(step)
          txcount = c(32768)
          if_op = scf.IfOp(is_leader)
          with ir.InsertionPoint(if_op.then_block):
            nvgpu.MBarrierArriveExpectTxOp(barrier_group, txcount, step)
            a_slice = memref.SubViewOp(
                ir.Type.parse("memref<128x64xf16, strided<[64, 1], offset: ?>, 3>"),
                a_smem, [step], [], [], [DYNAMIC, 0, 0], [1, *lhs_tile_shape], [1, 1, 1])
            # TODO: Coordinates seem wrong!!!
            nvgpu.TmaAsyncLoadOp(
                a_slice, barrier_group, a_tma_desc,
                coordinates=[arith.MulIOp(c(64), step), c(0)],
                mbarId=step)
            for b_start in (0, 64):
                b_slice = memref.SubViewOp(
                    ir.Type.parse("memref<64x64xf16, strided<[128, 1], offset: ?>, 3>"),
                    b_smem, [step], [], [], [DYNAMIC, b_start, 0], [1, *rhs_tma_shape], [1, 1, 1])
                nvgpu.TmaAsyncLoadOp(
                    b_slice, barrier_group, b_tma_desc,
                    coordinates=[c(b_start), arith.MulIOp(c(64), step)],
                    mbarId=step)
            scf.YieldOp([])

        fetch(0)
        fetch(1)
        acc = nvgpu.WarpgroupMmaInitAccumulatorOp(acc_ty).result

        for_op = scf.ForOp(c(0), c(2), c(1), [acc])
        with ir.InsertionPoint(for_op.body):
          i = for_op.induction_variable
          (carry_acc,) = for_op.inner_iter_args
          ticks = c(10000000)
          nvgpu.MBarrierTryWaitParityOp(barrier_group, c(0), ticks, mbarId=i)
          a_slice = memref.SubViewOp(
              ir.Type.parse("memref<128x64xf16, strided<[64, 1], offset: ?>, 3>"),
              a_smem, [i], [], [], [DYNAMIC, 0, 0], [1, 128, 64], [1, 1, 1])
          b_slice = memref.SubViewOp(
              ir.Type.parse("memref<64x128xf16, strided<[128, 1], offset: ?>, 3>"),
              b_smem, [i], [], [], [DYNAMIC, 0, 0], [1, 64, 128], [1, 1, 1])
          da = nvgpu.WarpgroupGenerateDescriptorOp(
              ir.Type.parse("!nvgpu.warpgroup.descriptor<tensor=memref<128x64xf16, 3>>"),
              a_slice, a_tma_desc)
          db = nvgpu.WarpgroupGenerateDescriptorOp(
              ir.Type.parse("!nvgpu.warpgroup.descriptor<tensor=memref<64x128xf16, 3>>"),
              b_slice, b_tma_desc)
          new_acc = nvgpu.WarpgroupMmaOp(acc.type, da, db, carry_acc, transposeB=True)
          scf.YieldOp(new_acc)
        acc = for_op.result
        """
        nvvm.wgmma.wait.group.sync.aligned 0  // Step 7. Wait all to finish mma
        """
        acc_smem = memref.GetGlobalOp(acc_smem_global.type_.value, "accSmem")
        acc_smem = memref.ReinterpretCastOp(
            ir.MemRefType.get((128, 128), f32, memory_space=smem),
            acc_smem, [], [], [], [0], [128, 128], [128, 1])
        nvgpu.WarpgroupMmaStoreOp(acc, acc_smem)

        warp = arith.DivUIOp(tidx, c(32))
        within_warp = arith.RemUIOp(tidx, c(32))
        off =  arith.MulIOp(within_warp, c(4))
        for_op = scf.ForOp(warp, c(128), c(4))
        with ir.InsertionPoint(for_op.body):
          acc_part = vector.LoadOp(ir.VectorType.get((4,), f32), acc_smem, [for_op.induction_variable, off])
          vector.StoreOp(acc_part, c_device, [for_op.induction_variable, off])
          scf.YieldOp([])

        gpu.TerminatorOp()
      gpu.MemcpyOp(token_ty, [token], c_host, c_device)
      gpu.WaitOp(token_ty, [token])

  print(m)
  main.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
  m.operation.verify()

  #pass_manager = PassManager.parse("builtin.module(convert-nvgpu-to-nvvm,gpu-kernel-outlining{data-layout-str=},convert-linalg-to-loops,convert-vector-to-scf{full-unroll=false lower-tensors=false target-rank=1},convert-scf-to-cf,convert-nvvm-to-llvm,convert-vector-to-llvm{enable-amx=false enable-arm-neon=false enable-arm-sve=false enable-x86vector=false force-32bit-vector-indices=true reassociate-fp-reductions=false},convert-math-to-llvm{approximate-log1p=true},finalize-memref-to-llvm{index-bitwidth=0 use-aligned-alloc=false use-generic-functions=false},convert-func-to-llvm{index-bitwidth=0 use-bare-ptr-memref-call-conv=false},expand-strided-metadata,nvvm-attach-target{O=3 chip=sm_90a fast=false features=+ptx80 ftz=false  module= triple=nvptx64-nvidia-cuda},lower-affine,convert-arith-to-llvm{index-bitwidth=0},convert-index-to-llvm{index-bitwidth=64},canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=true test-convergence=false top-down=true},cse,gpu.module(strip-debuginfo),gpu.module(convert-gpu-to-nvvm{has-redux=false index-bitwidth=64 use-bare-ptr-memref-call-conv=false}),gpu.module(canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=true test-convergence=false top-down=true}),gpu.module(cse),gpu.module(reconcile-unrealized-casts),gpu-to-llvm{gpu-binary-annotation=gpu.binary use-bare-pointers-for-host=false use-bare-pointers-for-kernels=false},gpu-module-to-binary{format=fatbin},canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=true test-convergence=false top-down=true},cse,reconcile-unrealized-casts)")
  passes = [
    "convert-nvgpu-to-nvvm",
    "gpu-kernel-outlining{data-layout-str=}",
    # "convert-linalg-to-loops",
    "convert-vector-to-scf{full-unroll=false lower-tensors=false target-rank=1}",
    "convert-scf-to-cf",
    "convert-nvvm-to-llvm",
    "convert-vector-to-llvm{enable-amx=false enable-arm-neon=false enable-arm-sve=false enable-x86vector=false force-32bit-vector-indices=true reassociate-fp-reductions=false}",
    "convert-math-to-llvm{approximate-log1p=true}",
    "finalize-memref-to-llvm{index-bitwidth=0 use-aligned-alloc=false use-generic-functions=false}",
    "convert-func-to-llvm{index-bitwidth=0 use-bare-ptr-memref-call-conv=false}",
    "expand-strided-metadata",
    "nvvm-attach-target{O=3 chip=sm_90a fast=false features=+ptx80 ftz=false  module= triple=nvptx64-nvidia-cuda}",
    "lower-affine",
    "convert-arith-to-llvm{index-bitwidth=0}",
    "convert-index-to-llvm{index-bitwidth=64}",
    "canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=true test-convergence=false top-down=true}",
    "cse",
    "gpu.module(strip-debuginfo)",
    "gpu.module(convert-gpu-to-nvvm{has-redux=false index-bitwidth=64 use-bare-ptr-memref-call-conv=false})",
    "gpu.module(canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=true test-convergence=false top-down=true})",
    "gpu.module(cse)",
    "gpu.module(reconcile-unrealized-casts)",
    "gpu-to-llvm{gpu-binary-annotation=gpu.binary use-bare-pointers-for-host=false use-bare-pointers-for-kernels=false}",
    "gpu-module-to-binary{format=fatbin}",
    "canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=true test-convergence=false top-down=true}",
    "cse",
    "reconcile-unrealized-casts",
  ]
  pass_manager = PassManager.parse(f"builtin.module({','.join(passes)})")
  pass_manager.run(m.operation)

  # ir._GlobalDebug.flag = True
  # pass_manager = PassManager.parse("builtin.module(gpu-module-to-binary{format=assembly})")
  # pass_manager.run(m.operation)
  # print(m)

  m.operation.verify()
  runtime_path = pathlib.Path(jaxlib.__file__).parent / 'cuda' / 'libmlir_cuda_runtime.so'
  assert runtime_path.exists()
  print(runtime_path)
  engine = ExecutionEngine(m, opt_level=3, shared_libs=[str(runtime_path)], enable_object_dump=False)
  # print(engine)
  print(engine.invoke("main"))
  # engine.invoke()
  print("OK")
