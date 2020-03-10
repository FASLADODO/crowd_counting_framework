from models import CompactCNN, AttnCanAdcrowdNetSimpleV3, CompactDilatedCNN, DefDilatedCCNN
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
    # print("dilate ccnn")
    # dcnn1 = CompactDilatedCNN()
    # print(summary(dcnn1, (3, 512, 512)))
    # print("=============================================================================")
    print("dilate ccnn")
    dcnn2 = DefDilatedCCNN()
    print(summary(dcnn2, (3, 512, 512)))
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