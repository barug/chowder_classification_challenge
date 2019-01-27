import torch 
from torch import nn

def print_forward_metrics(model, xb):
    model.eval()
    remove_handles = []

    def apply_hook(m):
        def forward_metrics(module, input, output):
            print("------------------------------------")
            print(module)
            print("output min : " + str(output.min().item()))
            print("output max : " + str(output.max().item()))
            print("output mean : " + str(output.mean().item()))
            print("output std : " + str(output.std().item()))
            
        if isinstance(m, (nn.Conv1d, nn.Linear, nn.Sigmoid)):
            remove_handles.append(m.register_forward_hook(forward_metrics))

    model.apply(apply_hook)
    with torch.no_grad():
        model(xb)

    for rm_hook in remove_handles:
        rm_hook.remove()

def print_backward_metrics(model, loss_func, xb, yb):
    model.eval()
    remove_handles = []

    def apply_hook(m):
        def backward_metrics(module, grad_input, output):
            print("------------------------------------")
            print(module)
            #print(output)
            print("gradient min : " + str(output.min().item()))
            print("gradient max : " + str(output.max().item()))
            print("gradient mean : " + str(output.mean().item()))
            print("gradient std : " + str(output.std().item()))
            print(grad_input)
            # print("gradient input min : " + str(grad_input.min().item()))
            # print("gradient input : " + str(grad_input.max().item()))
            # print("gradient mean : " + str(grad_input.mean().item()))
            # print("gradient std : " + str(grad_input.std().item()))
            
        if isinstance(m, (nn.Conv1d, nn.Linear, nn.Sigmoid)):
            remove_handles.append(m.register_forward_hook(backward_metrics))

    model.apply(apply_hook)
    y_pred = model(xb)
    loss = loss_func(y_pred, yb)
    loss.backward()

    for rm_hook in remove_handles:
        rm_hook.remove()

def print_weights_metrics(model):
        def print_lyr_weights(m):
            print(m)
            print("weights min : " + str(m.weight.data.min().item()))
            print("weights max : " + str(m.weight.data.max().item()))
            print("weights mean : " + str(m.weight.data.mean().item()))
            print("weights std : " + str(m.weight.data.std().item()))
        model.apply(print_lyr_weights)