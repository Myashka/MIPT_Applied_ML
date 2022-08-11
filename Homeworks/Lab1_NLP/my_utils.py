from pyexpat import model
import sched
from tokenize import String
import torch
from torch import load, save
import matplotlib.pyplot as plt

from IPython.display import clear_output
from torch.nn.utils import clip_grad_norm_

import time
from tqdm.auto import trange

from nltk.translate.bleu_score import corpus_bleu

import warnings
warnings.filterwarnings('ignore')

GOOGLE_PATH = '/content/drive/MyDrive/Colab Notebooks/NMT/'


def save_checkpoint(checkpoint_path: str, model, optimizer, lr_scheduler, epoch: int, global_step: int, colab: bool = False,
                    is_log: bool = False):
    """
    Save the current state of the model and optimizer to your path

    :Params:
    ----------
        chechpoint_path: String
            The path to the chechpoint. "*.pth"

        model: torch.nn.Module
            The model with state you want to save.

        optimizer: torch.optim.Optimizer
            The optimizer with state you want to save.

        lr_scheduler:
            Learning rate scheduler.

        epoch: int
            Number of epoch.

        global_step: int
            Number of steps.
    ----------
    """
    if lr_scheduler is not None:
        state = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'lr_sched': lr_scheduler,
            'global_step': global_step}
    else:
        state = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'global_step': global_step}
    if colab:
        save(state, GOOGLE_PATH+checkpoint_path)
    else:
        save(state, checkpoint_path)
    if is_log:
        print('Model saved to %s' % checkpoint_path)


def load_checkpoint(checkpoint_path: str, model, optimizer, device, lr_scheduler=None, colab: bool = True):
    """
    Load a state of the model and optimizer from your path

    :Params:
    ----------
        chechpoint_path: String
            The path to the chechpoint.

        model: torch.nn.Module
            The model which state you want load to.

        optimizer: torch.optim.Optimizer
            The optimizer which state you want load to.

        lr_scheduler:
            Learning rate scheduler.
    ----------

    :Returns:
    ----------
        global_step: int
            Number of steps.

        epoch: int
            Number of epoch.
    ----------
    """
    if colab:
        state = load(GOOGLE_PATH+checkpoint_path, map_location=device)
        state_cpu = load(GOOGLE_PATH+checkpoint_path)
    else:
        state = load(checkpoint_path, map_location=device)
        state_cpu = load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state_cpu['optimizer'])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(state_cpu['lr_sched'])

    epoch = state_cpu['epoch']
    global_step = state_cpu['global_step']
    print('model loaded from %s' % checkpoint_path)
    return global_step, epoch


def eval_bleu_score(model, trg_vocab, dataloader, device):
    """
    Evaluate a BLEU score of the model

    :Params:
    ----------
        model:
            The path to the chechpoint.

        trg_vocab: torch.nn.Module
            Target vocab.

        dataloader:
            Dataloader to eval.

        device:
            Device.
    ----------

    :Return: float
    ----------
        BLEU score.
    ----------
    """
    sos_token, eos_token, pad_token = "<sos>", "<eos>", "<pad>"
    specials = [sos_token, eos_token, pad_token]

    references, hypotheses = [], []
    trg_itos = trg_vocab.get_itos()
    with torch.no_grad():
        for src, trg in dataloader:
            output = model(src.to(device), trg.to(
                device), teacher_forcing_ratio=0)
            output = output.cpu().numpy().argmax(axis=2)

            for i in range(trg.shape[1]):
                reference = trg[:, i]
                reference_tokens = [trg_itos[id_] for id_ in reference]
                reference_tokens = [
                    tok for tok in reference_tokens if tok not in specials]
                references.append(reference_tokens)

                hypothesis = output[:, i]
                hypothesis_tokens = [trg_itos[id_] for id_ in hypothesis]
                hypothesis_tokens = [
                    tok for tok in hypothesis_tokens if tok not in specials]
                hypotheses.append(hypothesis_tokens)

    # corpus_bleu works with multiple references
    bleu = corpus_bleu([[ref] for ref in references], hypotheses)
    print(f"Your model shows BLEU of {100 * bleu:.1f}")
    return bleu


class Trainer:
    def __init__(self,
                 model,
                 criterion,
                 optimizer,
                 device,
                 train_dataloader,
                 val_dataloader,
                 model_name: str,
                 writer,
                 lr_scheduler=None,
                 colab: bool = True) -> None:

        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model_name = model_name
        self.writer = writer
        self.colab = colab

    def train(self, n_epochs, resume, eval_bleu: bool = False, trg_vocab=None):

        assert (eval_bleu == True & (trg_vocab is not None)), "To eval BLEU score you have to put trg vocab."

        global_step = 0
        epochs_done = 0

        best_loss = torch.inf
        best_bleu = 0
        clip = 1
        if resume:
            global_step, epochs_done = load_checkpoint(
                f'last_state_{self.model_name}.pth', self.model, self.optimizer, self.device, self.lr_scheduler, self.colab)

        for epoch in trange(n_epochs, desc='Epochs'):
            start = time.time()
            self.model.train()
            train_loss = 0
            for src, trg in self.train_dataloader:

                # zero_grad
                self.optimizer.zero_grad()

                src, trg = src.to(self.device), trg.to(self.device)
                output = self.model(src, trg)

                output = output.view(-1, output.shape[-1])
                trg = trg[1:].view(-1)

                # compute loss
                loss = self.criterion(output, trg)
                loss.backward()

                # optimizer step
                clip_grad_norm_(self.model.parameters(), clip)
                self.optimizer.step()

                train_loss += loss.item()
                # loss_history.append(loss.item())

                self.writer.add_scalar(
                    "Training/loss", loss.item(), global_step)
                global_step += 1

                # if len(loss_history) % 10 == 0:
                #     clear_output(wait=True)

                #     plt.figure(figsize=(15, 5))

                #     plt.subplot(121)
                #     plt.plot(loss_history)
                #     plt.xlabel("step")

                #     plt.subplot(122)
                #     plt.plot(train_loss_history, label="train loss")
                #     plt.plot(val_loss_history, label="val loss")
                #     plt.xlabel("epoch")
                #     plt.legend()

                #     plt.show()

            train_loss /= len(self.train_dataloader)
            # train_loss_history.append(train_loss)
            if eval_bleu:
                val_loss, bleu_score = self.eval_model(trg_vocab, True)
                self.writer.add_scalar(
                    "Evaluation/val_bleu", bleu_score, epochs_done)

            else:
                val_loss = self.eval_model()
            self.writer.add_scalars(
                "Evaluation", {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, epochs_done)

            # val_loss_history.append(val_loss)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(val_loss)

            epoch_time = int((time.time() - start) % 60)
            epochs_done += 1
            print(f'Epoch №{epochs_done} with {epoch_time} sec')

            if eval_bleu:
                if bleu_score > best_bleu:
                    best_bleu = bleu_score
                    save_checkpoint(f'best_{self.model_name}.pth',
                                    self.model, self.optimizer, self.lr_scheduler, epochs_done, global_step, self.colab, True)
            else:
                if val_loss < best_loss:
                    best_loss = val_loss
                    save_checkpoint(f'best_{self.model_name}.pth',
                                    self.model, self.optimizer, self.lr_scheduler, epochs_done, global_step, self.colab, True)

            save_checkpoint(f'last_state_{self.model_name}.pth',
                            self.model, self.optimizer, self.lr_scheduler, epochs_done, global_step, self.colab)
            
            self.writer.flush()

        self.writer.close()

    def eval_model(self, trg_vocab=None, compute_bleu=False):
        if compute_bleu:
            sos_token, eos_token, pad_token = "<sos>", "<eos>", "<pad>"
            specials = [sos_token, eos_token, pad_token]

            references, hypotheses = [], []
            trg_itos = trg_vocab.get_itos()
        val_loss = 0
        self.model.eval()
        with torch.no_grad():
            for src, trg in self.val_dataloader:
                src, trg = src.to(self.device), trg.to(self.device)
                output = self.model(src, trg, teacher_forcing_ratio=0)
                
                #___________Compute BLEU___________
                if compute_bleu:
                    output_for_blue = output.cpu().numpy().argmax(axis=2)

                    for i in range(trg.shape[1]):
                        reference = trg[:, i]
                        reference_tokens = [trg_itos[id_] for id_ in reference]
                        reference_tokens = [
                            tok for tok in reference_tokens if tok not in specials]
                        references.append(reference_tokens)

                        hypothesis = output_for_blue[:, i]
                        hypothesis_tokens = [trg_itos[id_]
                                             for id_ in hypothesis]
                        hypothesis_tokens = [
                            tok for tok in hypothesis_tokens if tok not in specials]
                        hypotheses.append(hypothesis_tokens)
                    
                    # corpus_bleu works with multiple references
                    bleu = corpus_bleu([[ref]
                                       for ref in references], hypotheses)

                #___________Compute val_loss___________                
                output = output.view(-1, output.shape[-1])

                trg = trg[1:].view(-1)

                loss = self.criterion(output, trg)

                val_loss += loss.item()

        val_loss /= len(self.val_dataloader)
        if compute_bleu:
            return val_loss, bleu*100
        else:
            return val_loss
