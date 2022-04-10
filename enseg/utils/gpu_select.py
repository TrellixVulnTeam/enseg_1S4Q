from functools import reduce


def get_available_gpu(default):
    number = len(default)
    try:
        import pynvml
    except:
        print("not pynvml")
        return list(range(len(default)))

    pynvml.nvmlInit()
    info = []
    for i in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        info.append([i, meminfo.free])
    info = list(sorted(info, key=lambda x: x[1], reverse=True))

    def to_str(obj):
        return "{}:{:.1f}GB".format(obj[0], obj[1] / 1024 / 1024 / 1024)

    print("select gpu:{}".format(" ".join(to_str(i) for i in info[:number])))
    return [i[0] for i in info[:number]]


if __name__ == "__main__":
    print(get_available_gpu([1]))
