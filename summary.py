import argparse
from calflops import calculate_flops
from nets.DPAF import DPAF

def parse_opt():
    parser = argparse.ArgumentParser(description="Analysis with calflops")

    parser.add_argument('--input_shape', type=int, nargs='+', default=[224, 224],
                        help='Input image shape: [Height, Width]')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='Number of classes')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    opt = parse_opt()

    # 初始化模型
    ours_model = DPAF(num_classes=opt.num_classes)

    # calflops 接收的 input_shape 需要包含 Batch 维度，这里设 Batch=1
    input_shape = (1, 3, opt.input_shape[0], opt.input_shape[1])
    
    # 调用 calculate_flops
    flops, macs, params = calculate_flops(model=ours_model, 
                                          input_shape=input_shape,
                                          print_results=True)
    
    print(f"最终统计 -> FLOPs: {flops}, Params: {params}")
