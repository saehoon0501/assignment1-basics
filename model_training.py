import torch
import time
import wandb
from cs336_basics.transformer import TransformerLM
from cs336_basics.loss_function import cross_entropy
from cs336_basics.optimizer import AdamW

# --- Hyperparameters ---
config = {
    # Model params
    "vocab_size": 10000,
    "context_length": 256,
    "d_model": 512,
    "num_layers": 4,
    "num_heads": 16,
    "d_ff": 1344,
    "rope_theta": 10000.0,

    # Training params
    "batch_size": 32,
    "learning_rate": 1e-3,
    "weight_decay": 1e-2,
    "betas": (0.9, 0.999),
    "eps": 1e-3,
    "num_epochs": 10,
    "log_interval": 10, # Log every 10 gradient steps
}

def get_dummy_weights(cfg):
    """
    Creates a dummy weights dictionary for the TransformerLM based on the config.
    """
    weights = {
        'token_embeddings.weight': torch.randn(cfg['vocab_size'], cfg['d_model']),
        'ln_final.weight': torch.randn(cfg['d_model']),
        'lm_head.weight': torch.randn(cfg['d_model'], cfg['vocab_size']),
    }
    for i in range(cfg['num_layers']):
        weights[f'layers.{i}.ln1.weight'] = torch.randn(cfg['d_model'])
        weights[f'layers.{i}.ln2.weight'] = torch.randn(cfg['d_model'])
        weights[f'layers.{i}.attn.q_proj.weight'] = torch.randn(cfg['d_model'], cfg['d_model'])
        weights[f'layers.{i}.attn.k_proj.weight'] = torch.randn(cfg['d_model'], cfg['d_model'])
        weights[f'layers.{i}.attn.v_proj.weight'] = torch.randn(cfg['d_model'], cfg['d_model'])
        weights[f'layers.{i}.attn.output_proj.weight'] = torch.randn(cfg['d_model'], cfg['d_model'])
        weights[f'layers.{i}.ffn.w1.weight'] = torch.randn(cfg['d_model'], cfg['d_ff'])
        weights[f'layers.{i}.ffn.w2.weight'] = torch.randn(cfg['d_ff'], cfg['d_model'])
        weights[f'layers.{i}.ffn.w3.weight'] = torch.randn(cfg['d_model'], cfg['d_ff'])
    return weights


def main():
    # 1. Initialize W&B run in anonymous mode
    # No login required. A private link will be generated for you.
    wandb.init(
        project="cs336-assignment1-basics", 
        config=config, 
        anonymous="allow"
    )
    
    # Use the config from wandb for consistency
    cfg = wandb.config

    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Set up model, optimizer, and data
    dummy_weights = get_dummy_weights(cfg)
    model = TransformerLM(
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        rope_theta=cfg.rope_theta,
        weights=dummy_weights,
    ).to(device)

    optimizer = AdamW(
        model.parameters(), 
        weight_decay=cfg.weight_decay, 
        betas=cfg.betas,
        eps=cfg.eps,
        lr=cfg.learning_rate
    )

    # Create a dummy dataset and dataloader
    train_data = torch.randint(0, cfg.vocab_size, (1000, cfg.context_length))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=cfg.batch_size)
    
    # 3. Training Loop
    start_time = time.time()
    wandb.watch(model, log="all", log_freq=cfg.log_interval)

    for epoch in range(cfg.num_epochs):
        for i, batch in enumerate(train_loader):
            model.train()
            
            inputs = batch[:, :-1].to(device)
            targets = batch[:, 1:].to(device)

            logits = model(inputs)
            
            loss = cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 4. Log to W&B
            if (i + 1) % cfg.log_interval == 0:
                elapsed_time = time.time() - start_time
                current_step = epoch * len(train_loader) + i
                
                tokens_processed = (current_step + 1) * cfg.batch_size * (cfg.context_length - 1)
                tokens_per_second = tokens_processed / elapsed_time if elapsed_time > 0 else 0

                # Log a dictionary of metrics
                wandb.log({
                    "epoch": epoch,
                    "train_loss": loss.item(),
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "tokens_per_second": tokens_per_second,
                    "wall_time": elapsed_time,
                })


                print(f"Epoch [{epoch+1}/{cfg.num_epochs}], Step [{current_step}], Loss: {loss.item():.4f}")

    # 5. Finish the W&B run
    wandb.finish()
    print("Training finished. Run logged to Weights & Biases.")

if __name__ == '__main__':
    main()
