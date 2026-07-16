import torch
import math

from finetuning import prepare_batch

# This method is used by HuggingFace to help not having any problem during the DDP
# For example for the padding of the validation set...


class SequentialDistributedSampler(torch.utils.data.Sampler):
    """
    Distributed evaluation sampler: each rank receives a contiguous 
    (continuous, without interruption or shuffle) 
    block of indices (no interleaving, unlike DistributedSampler). 
    The padding required to make the total size divisible by world_size 
    is added only at the end of the global sequence (repetition of 
    the first indices), thus entirely contained within the last block. 
    Two guarantees: 
    - All ranks receive EXACTLY the same number of 
    samples (num_samples), hence the same number of batches => 
    same number of forward calls => no broadcast_buffers desynchronization
    in DDP. 
    - By concatenating the gathered results in the order 
    of ranks and then truncating to len(dataset), we end up with 
    exactly the unique samples from the dataset, without duplication.
    """

    def __init__(self, dataset, world_size: int, rank: int):
        self.dataset = dataset
        self.world_size = world_size
        self.rank = rank
        self.num_samples = int(math.ceil(len(dataset) / world_size))
        self.total_size = self.num_samples * world_size

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # padding at the end of the sequence
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size
        start = self.rank * self.num_samples
        end = start + self.num_samples
        return iter(indices[start:end])

    def __len__(self):
        return self.num_samples


@torch.no_grad()
def evaluate(model, loader, device, cfg, is_distributed, num_val_samples: int):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        sar, optical, labels = prepare_batch(batch, device)
        with torch.amp.autocast(device_type="cuda", dtype=cfg.mixed_precision):
            logits = model(sar, optical)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    if is_distributed:
        # Toutes les tailles locales sont désormais identiques (num_samples
        # fixe par construction du sampler), donc all_gather standard est
        # sûr (il exige des tensors de même shape sur tous les ranks).
        gathered_preds = [torch.zeros_like(all_preds) for _ in range(
            torch.distributed.get_world_size())]
        gathered_labels = [torch.zeros_like(all_labels) for _ in range(
            torch.distributed.get_world_size())]
        torch.distributed.all_gather(gathered_preds, all_preds)
        torch.distributed.all_gather(gathered_labels, all_labels)

        # Concaténation dans l'ordre des ranks (0,1,2,...) : reconstitue la
        # séquence globale paddée, où le padding est garanti tout à la fin.
        all_preds = torch.cat(gathered_preds, dim=0)[:num_val_samples]
        all_labels = torch.cat(gathered_labels, dim=0)[:num_val_samples]

    oa, aa = compute_oa_aa(all_preds, all_labels)
    return oa, aa
