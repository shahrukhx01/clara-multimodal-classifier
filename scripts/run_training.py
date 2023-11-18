import argparse
import torch

from clara_mm_classifiers.models.clara_audio_linear_probe import CLARAAudioLinearProbe
from clara_mm_classifiers.datasets.ravdess_data_module import RavdessDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger as PlMLFlowLogger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model_path", type=str, help="Path to model")

    # parser.add_argument('--task', type=str, choices=['texts', 'gender', 'emotion', 'age', 'sounds', 'speech'], help='Task to run')
    parser.add_argument('--dataset_path', type=str, required=True, help='Full path to the dataset pickle on disk.')
    # parser.add_argument('--root_cfg_path', type=str, default='./config/', help='root path to config files')
    # parser.add_argument('--top_k', type=int, default=[1,5,10], help='Top k metrics to use')
    # parser.add_argument('--batch_size', type=int, default=8, help='Dataloader batch size')
    # parser.add_argument('--num_workers', type=int, default=12, help='Dataloader number of workers')
    args = parser.parse_args()
    model = CLARAAudioLinearProbe(num_classes=8, clara_checkpoint_path=args.model_path, clara_map_location=device)
    dataset_config = {"dataset_path": args.dataset_path}
    datamodule = RavdessDataModule(config=dataset_config)

    trainer = Trainer(max_epochs=1, accelerator="cuda", precision=16)
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)
