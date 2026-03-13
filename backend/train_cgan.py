from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from generator import IMAGE_SIZE, LATENT_DIM, NUM_CLASSES, Generator


class Discriminator(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        image_dim = IMAGE_SIZE * IMAGE_SIZE
        self.model = nn.Sequential(
            nn.Linear(image_dim + num_classes, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        flattened = images.view(images.size(0), -1)
        one_hot_labels = F.one_hot(labels, num_classes=NUM_CLASSES).float()
        discriminator_input = torch.cat((flattened, one_hot_labels), dim=1)
        return self.model(discriminator_input)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Conditional GAN on MNIST")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument(
        "--generator-path",
        type=Path,
        default=Path(__file__).resolve().parent / "generator_cgan.pth",
    )
    parser.add_argument(
        "--discriminator-path",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "discriminator_cgan.pth",
    )
    return parser.parse_args()


def train() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    dataset = datasets.MNIST(
        root=str(Path(__file__).resolve().parent / "data"),
        train=True,
        transform=transform,
        download=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    adversarial_loss = nn.BCELoss()

    optimizer_g = optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(
        discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999)
    )

    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()

        running_g_loss = 0.0
        running_d_loss = 0.0

        for real_images, real_labels in dataloader:
            real_images = real_images.to(device)
            real_labels = real_labels.to(device)
            batch_size = real_images.size(0)

            valid_targets = torch.ones((batch_size, 1), device=device)
            fake_targets = torch.zeros((batch_size, 1), device=device)

            optimizer_d.zero_grad()

            real_pred = discriminator(real_images, real_labels)
            d_real_loss = adversarial_loss(real_pred, valid_targets)

            sampled_labels = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)
            noise = torch.randn(batch_size, LATENT_DIM, device=device)
            with torch.no_grad():
                fake_images = generator(noise, sampled_labels)

            fake_pred = discriminator(fake_images, sampled_labels)
            d_fake_loss = adversarial_loss(fake_pred, fake_targets)

            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_d.step()

            optimizer_g.zero_grad()

            sampled_labels = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)
            noise = torch.randn(batch_size, LATENT_DIM, device=device)
            generated_images = generator(noise, sampled_labels)
            generator_pred = discriminator(generated_images, sampled_labels)
            g_loss = adversarial_loss(generator_pred, valid_targets)

            g_loss.backward()
            optimizer_g.step()

            running_d_loss += d_loss.item()
            running_g_loss += g_loss.item()

        avg_d_loss = running_d_loss / len(dataloader)
        avg_g_loss = running_g_loss / len(dataloader)
        print(f"Epoch [{epoch}/{args.epochs}] D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            args.generator_path.parent.mkdir(parents=True, exist_ok=True)
            args.discriminator_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(generator.state_dict(), args.generator_path)
            torch.save(discriminator.state_dict(), args.discriminator_path)
            print(
                f"Saved checkpoints at epoch {epoch}:\n"
                f"  Generator -> {args.generator_path}\n"
                f"  Discriminator -> {args.discriminator_path}"
            )


if __name__ == "__main__":
    train()