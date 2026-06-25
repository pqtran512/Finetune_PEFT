# example_numpy.py
import numpy as np

def main():
    print("=== Numpy Basic Example ===\n")
    
    # 1️⃣ Tạo mảng
    a = np.array([1, 2, 3, 4])
    b = np.array([5, 6, 7, 8])
    print("Array a:", a)
    print("Array b:", b)

    # 2️⃣ Cộng trừ nhân chia phần tử
    print("\nCộng a + b:", a + b)
    print("Trừ a - b:", a - b)
    print("Nhân a * b:", a * b)
    print("Chia a / b:", a / b)

    # 3️⃣ Trung bình, max, min
    print("\nTrung bình a:", np.mean(a))
    print("Max a:", np.max(a))
    print("Min a:", np.min(a))

    # 4️⃣ Ma trận
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    print("\nMatrix A:\n", A)
    print("Matrix B:\n", B)

    # Nhân ma trận
    C = np.dot(A, B)
    print("\nA dot B:\n", C)

    # 5️⃣ Chuẩn hóa mảng (min-max scaling)
    arr = np.array([10, 20, 30, 40, 50])
    arr_scaled = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    print("\nOriginal array:", arr)
    print("Scaled array (0-1):", arr_scaled)

    # 6️⃣ Random và reshape
    rand_arr = np.random.rand(6)  # 6 số ngẫu nhiên 0-1
    rand_arr_reshaped = rand_arr.reshape(2, 3)
    print("\nRandom array 1D:", rand_arr)
    print("Reshaped 2x3:\n", rand_arr_reshaped)

if __name__ == "__main__":
    main()
