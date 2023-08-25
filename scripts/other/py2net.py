import torch
import click
import segmentation_models_pytorch as smp
# import onnx
from pathlib import Path
# from onnx_tf.backend import prepare

@click.command()
@click.option('--model_path', type=Path)
@click.option('--backbone', type=str)
@click.option('--n_classes', type=int)
def main(model_path, backbone, n_classes):
        
    model = smp.Unet(encoder_name=backbone, classes=n_classes, activation=None)
    model.load_state_dict(torch.load(model_path / 'model_last_statedict.pt'))

    torch.save(model.state_dict(), model_path / 'model_std.pt', _use_new_zipfile_serialization=False)

    # model_scripted = torch.jit.script(model) # Export to TorchScript
    # model_scripted.save('model_scripted.pt')
    # dummy_input = torch.randn(10, 3, 224, 224)
    # # torch.onnx.export(model, dummy_input, model_path / "model.net")
    
    # # model = onnx.load('mnist.onnx')
    # tf_rep = prepare(model) 
    
    # tf_rep.export_graph(model_path / 'model.net')

if '__main__' == __name__:
    main()