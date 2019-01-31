import torch 
from torch import nn
from sklearn import metrics


def compute_forward_metrics(model, xb):
    model.eval()
    remove_handles = []

    def apply_hook(m):
        def forward_metrics(m, lay_input, output):
            act_metrics = {}
            act_metrics["min"] = output.min().item()
            act_metrics["max"] = output.max().item()
            act_metrics["mean"] = output.mean().item()
            act_metrics["std"] = output.std().item()
            act_metrics["output"] = output
            act_metrics["input"] = lay_input
            
            if not hasattr(m, 'my_metrics'):
                m.my_metrics = {}
            m.my_metrics["forward"] = act_metrics
        #if isinstance(m, (nn.Conv1d, nn.Linear, nn.LeakyReLU)):
        remove_handles.append(m.register_forward_hook(forward_metrics))

    model.apply(apply_hook)
    with torch.no_grad():
        model(xb)

    for rm_hook in remove_handles:
        rm_hook.remove()

def compute_backward_metrics(model, loss_func, xb, yb):
    model.eval()
    remove_handles = []

    def apply_hook(m):
        def backward_metrics(m, grad_input, output):
            grad_metrics = {}
            grad_metrics["min"] = output.min().item()
            grad_metrics["max"] = output.max().item()
            grad_metrics["mean"] = output.mean().item()
            grad_metrics["std"] = output.std().item()
            
            if not hasattr(m, 'my_metrics'):
                m.my_metrics = {}
            m.my_metrics["grads"] = grad_metrics
        if isinstance(m, (nn.Conv1d, nn.Linear, nn.LeakyReLU)):
            remove_handles.append(m.register_forward_hook(backward_metrics))

    model.apply(apply_hook)
    y_pred = model(xb)
    loss = loss_func(y_pred, yb)
    loss.backward()

    for rm_hook in remove_handles:
        rm_hook.remove()

def compute_weights_metrics(model):
    def compute_lyr_weights(m):
        if isinstance(m, (nn.Conv1d, nn.Linear, nn.Sigmoid)):
            weights_metrics = {}
            weights_metrics["min"] = m.weight.data.min().item()
            weights_metrics["max"] = m.weight.data.max().item()
            weights_metrics["mean"] = m.weight.data.max().item()
            weights_metrics["std"] = m.weight.data.std().item()
            if not hasattr(m, 'my_metrics'):
                m.my_metrics = {}
            m.my_metrics['weights'] = weights_metrics
    model.apply(compute_lyr_weights)
    

def pred_metrics(labels_test, labels_pred):
    labels_testset = labels_test.astype(int)
    print("confusion matrix: \n" + str(metrics.confusion_matrix(labels_testset, labels_pred)))
    print("accuracy : " + str(metrics.accuracy_score(labels_test, labels_pred)))

