import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torchtext.data.metrics import bleu_score
import pytorch_lightning as pl
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.model.models import Transformer
from src.model.utils import LearningRateWarmupScheduler, TranslationLoss
from extras.constants import *


class TransformerModule(pl.LightningModule):
    def __init__(
        self,
        vocab_de_size,
        vocab_en_size,
        vocab_de_idx2token,
        vocab_en_idx2token,
        pad_idx,
        warmup_steps=4000,
        label_smoothing=0.1,
        d_model=512,
        max_seq_len=256,
        h=8,
        d_ff=2048,
        p_drop=0.1,
        N=6,
        betas=(0.9, 0.98),
        eps=1e-9,
        device="cuda",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = Transformer(
            vocab_de_size,
            vocab_en_size,
            d_model,
            max_seq_len,
            h,
            d_ff,
            p_drop,
            N,
            device=torch.device(device),
        )
        self.loss_fn_train = TranslationLoss(pad_idx, vocab_en_size, label_smoothing)
        self.loss_fn_val = TranslationLoss(pad_idx, vocab_en_size, label_smoothing=0)
        self.bleu_score = bleu_score

    def length_penalty(self, length, alpha=0.6):
        return ((5 + length) ** alpha) / ((5 + 1) ** alpha)

    def translator(self, src, beam_size=3):  # beam search
        max_output_length = src.shape[1] + 50
        src = src.to(torch.device(self.hparams.device))

        self.model.eval()

        with torch.no_grad():
            encoder_input_embedded = self.model.embedding_src(src)
            encoder_output = self.model.encoder(encoder_input_embedded)

            decoder_input = torch.tensor([[SPECIAL_TOKENS_IDX["SOS"]]]).to(
                torch.device(self.hparams.device)
            )
            scores = torch.Tensor([0.0]).to(torch.device(self.hparams.device))

            for i in range(max_output_length):
                decoder_input_embedded = self.model.embedding_tgt(decoder_input)
                decoder_output = self.model.decoder(
                    decoder_input_embedded, encoder_output
                )
                logits = self.model.linear(decoder_output)

                log_probs = F.log_softmax(logits[:, -1], dim=-1) / self.length_penalty(
                    i + 1
                )
                paths_EOS_reached = decoder_input[:, -1] == SPECIAL_TOKENS_IDX["EOS"]
                log_probs[paths_EOS_reached, :] = 0

                scores = scores.unsqueeze(1) + log_probs

                scores, indices = torch.topk(scores.view(-1), k=beam_size)
                beam_indices = torch.divide(
                    indices, self.hparams.vocab_en_size, rounding_mode="floor"
                )
                token_indices = torch.remainder(indices, self.hparams.vocab_en_size)

                next_decoder_input = []
                for beam_index, token_index in zip(beam_indices, token_indices):
                    prev_decoder_input = decoder_input[beam_index]

                    if prev_decoder_input[-1] == SPECIAL_TOKENS_IDX["EOS"]:
                        token_index = SPECIAL_TOKENS_IDX["EOS"]

                    token_index = torch.LongTensor([token_index]).to(
                        torch.device(self.hparams.device)
                    )
                    next_decoder_input.append(
                        torch.cat([prev_decoder_input, token_index])
                    )

                decoder_input = torch.vstack(next_decoder_input)

                if (decoder_input[:, -1] == SPECIAL_TOKENS_IDX["EOS"]).all():
                    break

                if i == 0:
                    encoder_output = encoder_output.expand(
                        beam_size, *encoder_output.shape[1:]
                    )

        transformer_output, _ = max(zip(decoder_input, scores), key=lambda x: x[1])
        transformer_output = transformer_output[1:].cpu().numpy()

        output_texts = []
        for token_idx in transformer_output:
            if token_idx == SPECIAL_TOKENS_IDX["EOS"]:
                break
            output_texts.append(self.hparams.vocab_en_idx2token[token_idx])

        return output_texts

    def forward(self, src, tgt):
        return self.model(src, tgt)

    def training_step(self, batch, batch_idx):
        src, tgt = batch

        logits = self(src, tgt[:, :-1])
        loss = self.loss_fn_train(logits, tgt[:, 1:])

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch

        logits = self(src, tgt[:, :-1])

        loss = self.loss_fn_val(logits, tgt[:, 1:])

        self.log("val_loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(), betas=self.hparams.betas, eps=self.hparams.eps
        )
        scheduler = {
            "scheduler": LearningRateWarmupScheduler(
                optimizer,
                warmup_steps=self.hparams.warmup_steps,
                d_model=self.hparams.d_model,
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    def load_average_ckpt(self, ckpt_dir):
        ckpt_paths = [ckpt_path for ckpt_path in ckpt_dir.iterdir()]
        model_weights = [torch.load(path)["state_dict"] for path in ckpt_paths]
        length = len(model_weights)

        for key in model_weights[0]:
            model_weights[0][key] = model_weights[0][key] / length
            for i in range(1, length):
                model_weights[0][key] += model_weights[i][key] / length

        self.load_state_dict(model_weights[0])
