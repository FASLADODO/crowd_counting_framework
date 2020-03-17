from models import CompactCNN, AttnCanAdcrowdNetSimpleV3, CompactDilatedCNN, DefDilatedCCNN, CompactCNNV2, CustomCNNv3, CustomCNNv4
from torchsummary import summary

def very_simple_param_count(model):
    result = sum([p.numel() for p in model.parameters()])
    return result

if __name__ == "__main__":
    # print("Compact CNN")
    # ccnn = CompactCNN()
    # print(ccnn)
    # print("-------------")
    # # print(summary(ccnn, (3, 128, 128)))  # we print twice to confirm trainable parameter independent with input size
    # print("-------------")
    # print(summary(ccnn, (3, 512, 512)))
    # print("simple count", very_simple_param_count(ccnn))
    # print("===========================================================================")
    print("ccnn")
    dcnn1 = CompactCNN()
    print(summary(dcnn1, (3, 512, 512), device="cpu"))
    print("=============================================================================")
    print("ccnn v2")
    dcnn2 = CompactCNNV2()
    print(summary(dcnn2, (3, 512, 512),  device="cpu"))
    print("==============================================")
    print("custom ccnn v3")
    customCCNNv3 = CustomCNNv3()
    print(summary(customCCNNv3, (3, 512, 512),  device="cpu"))
    print("=============================================================================")
    print("custom ccnn v4")
    customCCNNv4 = CustomCNNv4()
    print(summary(customCCNNv4, (3, 512, 512),  device="cpu"))
    print("=============================================================================")
    # print("simple_v3")
    # simplev3 = AttnCanAdcrowdNetSimpleV3()
    # print(simplev3)
    # print("-------------")
    # # print(summary(ccnn, (3, 128, 128)))  # we print twice to confirm trainable parameter independent with input size
    # print("-------------")
    # print(summary(simplev3, (3, 512, 512)))
    # print("simple count", very_simple_param_count(simplev3))
    #