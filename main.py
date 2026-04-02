import torch
import torch.nn as nn
import time

# ========== 设备信息 ==========
def print_device_info():
    print("=" * 50)
    print("🖥️  设备信息")
    print("=" * 50)
    print(f"PyTorch 版本:     {torch.__version__}")
    print(f"CUDA 可用:        {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本:        {torch.version.cuda}")
        print(f"GPU 名称:         {torch.cuda.get_device_name(0)}")
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"显存总量:         {total:.1f} GB")
    print()

# ========== 测试模型（类似 ResNet Block） ==========
class BenchmarkModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),   nn.BatchNorm2d(64),  nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),  nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ========== 矩阵乘法速度测试 ==========
def test_matmul(device, size=4096, repeat=50):
    print(f"📐 矩阵乘法测试 ({size}x{size}, 重复 {repeat} 次)")
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    # 预热
    for _ in range(3):
        torch.mm(a, b)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        torch.mm(a, b)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tflops = (2 * size**3 * repeat) / elapsed / 1e12
    print(f"  耗时: {elapsed:.3f}s  |  吞吐: {tflops:.2f} TFLOPS\n")
    return tflops


# ========== 前向传播速度测试 ==========
def test_forward(device, batch_size=64, repeat=100):
    print(f"⚡ 前向传播测试 (batch={batch_size}, 重复 {repeat} 次)")
    model = BenchmarkModel().to(device)
    model.eval()
    x = torch.randn(batch_size, 3, 64, 64, device=device)
    # 预热
    with torch.no_grad():
        for _ in range(5):
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeat):
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    fps = batch_size * repeat / elapsed
    print(f"  耗时: {elapsed:.3f}s  |  吞吐: {fps:.0f} samples/s\n")
    return fps


# ========== 训练速度测试（含反向传播） ==========
def test_training(device, batch_size=64, repeat=50):
    print(f"🔥 训练速度测试 (batch={batch_size}, 重复 {repeat} 次，含反向传播)")
    model = BenchmarkModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    x = torch.randn(batch_size, 3, 64, 64, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)
    # 预热
    for _ in range(3):
        optimizer.zero_grad()
        criterion(model(x), y).backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    fps = batch_size * repeat / elapsed
    print(f"  耗时: {elapsed:.3f}s  |  吞吐: {fps:.0f} samples/s\n")
    return fps


# ========== AMP 混合精度对比 ==========
def test_amp(device, batch_size=64, repeat=50):
    if device.type != "cuda":
        return
    print(f"⚡ AMP 混合精度对比 (batch={batch_size}, 重复 {repeat} 次)")
    model = BenchmarkModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss()
    x = torch.randn(batch_size, 3, 64, 64, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)

    # FP32
    for _ in range(3):
        optimizer.zero_grad(); criterion(model(x), y).backward(); optimizer.step()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        optimizer.zero_grad(); criterion(model(x), y).backward(); optimizer.step()
    torch.cuda.synchronize()
    t_fp32 = time.perf_counter() - start

    # AMP FP16
    for _ in range(3):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            loss = criterion(model(x), y)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            loss = criterion(model(x), y)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
    torch.cuda.synchronize()
    t_amp = time.perf_counter() - start

    speedup = t_fp32 / t_amp
    print(f"  FP32 耗时: {t_fp32:.3f}s  |  AMP FP16 耗时: {t_amp:.3f}s  |  加速比: {speedup:.2f}x\n")


# ========== 显存带宽测试 ==========
def test_memory_bandwidth(device, size_gb=1.0):
    if device.type != "cuda":
        return
    print(f"💾 显存带宽测试 (数据量 {size_gb} GB)")
    n = int(size_gb * 1024**3 / 4)  # float32
    a = torch.randn(n, device=device)
    b = torch.empty_like(a)
    # 预热
    b.copy_(a)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(10):
        b.copy_(a)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / 10

    bandwidth = size_gb / elapsed
    print(f"  带宽: {bandwidth:.1f} GB/s\n")


# ========== 主函数 ==========
def main():
    print_device_info()
    results = {}

    for dev_name in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
        device = torch.device(dev_name)
        print(f"{'='*50}")
        print(f"🚀 测试设备: {dev_name.upper()}")
        print(f"{'='*50}\n")

        results[dev_name] = {}
        results[dev_name]["matmul_tflops"] = test_matmul(device)
        results[dev_name]["forward_fps"]   = test_forward(device)
        results[dev_name]["train_fps"]     = test_training(device)
        if dev_name == "cuda":
            test_amp(device)
            test_memory_bandwidth(device, size_gb=15.0)

    # 加速比汇总
    if "cpu" in results and "cuda" in results:
        print("=" * 50)
        print("📊 GPU vs CPU 加速比汇总")
        print("=" * 50)
        for key, label in [("matmul_tflops", "矩阵乘法"), ("forward_fps", "前向传播"), ("train_fps", "训练速度")]:
            ratio = results["cuda"][key] / results["cpu"][key]
            print(f"  {label}: {ratio:.1f}x")
        print()

    if torch.cuda.is_available():
        used  = torch.cuda.memory_allocated(0)  / 1024**2
        total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"📦 显存占用: {used:.0f} MB / {total:.0f} MB")

def main2():
    device = torch.device("cuda")
    test_amp(device)
    test_memory_bandwidth(device, size_gb=9.0)
    if torch.cuda.is_available():
        used  = torch.cuda.memory_allocated(0)  / 1024**2
        total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"📦 显存占用: {used:.0f} MB / {total:.0f} MB")
if __name__ == "__main__":
    main2()