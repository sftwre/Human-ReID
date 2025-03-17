import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import models
from tqdm import tqdm


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_out")
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm1d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)


class ClassHead(nn.Module):
    def __init__(
        self,
        input_dim,
        num_class,
    ):
        """
        Classification head used to fine-tune the densenet backbone on the Market-1501 dataset
        by classifying image features into num_class identities.
        """

        super(ClassHead, self).__init__()

        bottleneck_size = 512

        self.add_block = nn.Sequential(
            nn.Linear(input_dim, bottleneck_size),
            nn.BatchNorm1d(bottleneck_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.5),
        )

        self.classifier = nn.Sequential(nn.Linear(bottleneck_size, num_class))

        self.add_block.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x


class ReIDModel(nn.Module):

    def __init__(self, num_class: int):
        """
        Re-Idenfication model with a densenet backbone and a classification head.

        Args:
            num_class: number of unique identities to classify
        """

        super().__init__()

        self.emb_dim = 1024

        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        densenet.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        densenet.fc = nn.Sequential()

        self.model = densenet
        self.classifier = ClassHead(self.emb_dim, num_class)

    def forward(self, x):
        x = self.model.features(x)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x


class ImgEmbedder(nn.Module):
    def __init__(self, model_path: str):
        """
        Serves to facilitate image retrieval by extracting
        discriminative image embeddings from a pre-trained Re-Identification model.

        Args:
           model_path (str): Path to pre-trained model weights
        """
        super().__init__()
        self.num_classes = 751
        self.emb_dim = 1024
        self.backbone = ReIDModel(self.num_classes).eval()

        # load pre-trained model weights
        self.backbone.load_state_dict(torch.load(model_path, map_location="cpu"))

        # remove classification layer to extract image embeddings from base layers
        self.backbone.model.fc = nn.Sequential()
        self.backbone.classifier = nn.Sequential()

        # place model on appropriate device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.backbone.to(self.device)

    @torch.no_grad()
    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Extracts un-normalized image embeddings from a given image tensor.
        Args:
            x : Image tensor
        Returns:
            img_emb : CPU-bound image embeddings
        """
        x = x.to(self.device)
        img_emb = self.backbone(x).cpu()
        return img_emb

    def to_embeddings(self, dataloader: data.DataLoader) -> torch.tensor:
        """
        Extracts embeddings from a collection of images and normalizes the embeddings.

        Args:
            dataloader: torch DataLoader with image dataset
        Returns:
            embs: CPU-bound Tensor of normalized embeddings with shape (N, emb_dim),
            where N is the number of samples in the dataset and emb_dim
            is the embedding dimension (1024)
        """

        # normalized image embeddings
        norm_embs = torch.FloatTensor()

        # extract embeddings from a batch of images, normazlize them, and merge results with norm_embs
        for imgs, _ in tqdm(dataloader):

            img_embs = self.forward(imgs)

            fnorm = torch.norm(img_embs, p=2, dim=1, keepdim=True)
            img_embs = img_embs.div(fnorm.expand_as(img_embs))
            norm_embs = torch.cat((norm_embs, img_embs), 0)

        return norm_embs
