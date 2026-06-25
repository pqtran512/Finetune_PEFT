def check_pytorch():
    print("=== PyTorch ===")
    try:
        import torch
        print("PyTorch version:", torch.__version__)
        cuda_available = torch.cuda.is_available()
        print("CUDA available:", cuda_available)

        if cuda_available:
            print("GPU count:", torch.cuda.device_count())
            print("Current device:", torch.cuda.current_device())
            print("Device name:", torch.cuda.get_device_name(0))

            # test tensor training on GPU
            x = torch.randn(1000, 1000).cuda()
            w = torch.randn(1000, 1000, requires_grad=True, device="cuda")
            y = x @ w
            loss = y.mean()
            loss.backward()
            print("PyTorch GPU training test: OK")
        else:
            print("PyTorch will use CPU")
    except Exception as e:
        print("PyTorch error:", e)


def check_tensorflow():
    print("\n=== TensorFlow ===")
    try:
        import tensorflow as tf
        print("TensorFlow version:", tf.__version__)
        gpus = tf.config.list_physical_devices("GPU")
        print("GPUs found:", gpus)

        if gpus:
            # test training on GPU
            with tf.device("/GPU:0"):
                a = tf.random.normal([1000, 1000])
                b = tf.random.normal([1000, 1000])
                c = tf.matmul(a, b)
            print("TensorFlow GPU training test: OK")
        else:
            print("TensorFlow will use CPU")
    except Exception as e:
        print("TensorFlow error:", e)


if __name__ == "__main__":
    check_pytorch()
    check_tensorflow()