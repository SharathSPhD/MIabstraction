"""Debug why models aren't learning bracket matching despite well-posed task."""
import torch
from miabstraction.data.algo import BracketMatchingDataset
from miabstraction.models import TinyTransformer

torch.manual_seed(0)

# Create a tiny training setup
ds_train = BracketMatchingDataset(vocab_size=8, seq_len=16, n_samples=100, seed=0)
train_seqs = torch.stack([ds_train[i] for i in range(len(ds_train))])

model = TinyTransformer(vocab=8, d_model=32, n_layers=2, n_heads=2)
model.to("cpu").train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

print("=== INITIAL STATE ===")
print(f"Train seqs shape: {train_seqs.shape}")
print(f"Sample seq: {train_seqs[0].tolist()}")

# Get mask info for first sample
seq, mask, correct = ds_train.get_with_mask(0)
ans_pos = mask.nonzero(as_tuple=True)[0].item()
print(f"Answer position: {ans_pos}, correct closing: {correct}")

# Single forward pass to understand what's happening
print("\n=== FORWARD PASS ===")
batch = train_seqs[:32, :-1]  # Remove last token for input
targets = train_seqs[:32, 1:]   # Shift by 1 for targets

logits = model(batch)
print(f"Input shape: {batch.shape}")
print(f"Logits shape: {logits.shape}")
print(f"Targets shape: {targets.shape}")

# Compute cross-entropy loss
loss = torch.nn.functional.cross_entropy(
    logits.reshape(-1, 8), targets.reshape(-1)
)
print(f"Cross-entropy loss (all tokens): {loss.item():.4f}")

# Compute loss ONLY at answer positions
answer_losses = []
for b in range(min(10, batch.shape[0])):
    seq, mask, correct = ds_train.get_with_mask(b)
    ans_pos = mask.nonzero(as_tuple=True)[0].item()
    if ans_pos < logits.shape[1]:
        # Loss at answer position
        pred_logits = logits[b, ans_pos, :]
        target = correct
        token_loss = torch.nn.functional.cross_entropy(
            pred_logits.unsqueeze(0), torch.tensor([target])
        )
        answer_losses.append(token_loss.item())
        pred = pred_logits.argmax().item()
        print(f"Sample {b}: ans_pos={ans_pos}, target={target}, pred={pred}, loss={token_loss.item():.4f}")

avg_ans_loss = sum(answer_losses) / len(answer_losses)
print(f"\nAverage loss at answer positions: {avg_ans_loss:.4f}")
print(f"ln(8) = {torch.tensor(8.0).log().item():.4f} (random chance)")
print(f"ln(2) = {torch.tensor(2.0).log().item():.4f} (optimal for 2 classes)")

# Train for a few steps
print("\n=== TRAINING FOR 10 STEPS ===")
for step in range(10):
    batch = train_seqs[:64, :-1]
    targets = train_seqs[:64, 1:]
    logits = model(batch)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 8), targets.reshape(-1)
    )
    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 3 == 0:
        print(f"Step {step}: loss={loss.item():.4f}")

# Evaluate
print("\n=== EVALUATION (RANDOM INITIALIZATION) ===")
model.eval()
with torch.no_grad():
    correct_at_ans = 0
    for b in range(min(20, len(ds_train))):
        seq, mask, expected = ds_train.get_with_mask(b)
        ans_pos = mask.nonzero(as_tuple=True)[0].item()

        logits_eval = model(seq[:-1].unsqueeze(0))
        if ans_pos < logits_eval.shape[1]:
            pred = logits_eval[0, ans_pos, :].argmax().item()
            if pred == expected:
                correct_at_ans += 1

    acc = correct_at_ans / 20
    print(f"Accuracy at answer positions (after 10 steps): {acc:.1%}")
