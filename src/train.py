
import torch
import time


def train(model, train_dataloader, val_dataloader, optimizer, scheduler, device, num_epochs):
    train_losses = []
    val_losses = []

    for n in range(num_epochs):
        model.train()
        train_loss = 0
        start_time = time.time()

        # Training loop
        for k, (mb_x, mb_m, mb_y) in enumerate(train_dataloader):
            optimizer.zero_grad()

            mb_x = mb_x.to(device)
            mb_m = mb_m.to(device)
            mb_y = mb_y.to(device)

            outputs = model(mb_x, attention_mask=mb_m, labels=mb_y)

            loss = outputs[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() / len(train_dataloader)

        # Evaluate the model on the validation set
        val_loss = evaluate(model, val_dataloader, device)
        val_losses.append(val_loss)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch {n + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}')

        train_losses.append(train_loss)

    return train_losses, val_losses


def evaluate(model, dataloader, device):
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for k, (mb_x, mb_m, mb_y) in enumerate(dataloader):
            mb_x = mb_x.to(device)
            mb_m = mb_m.to(device)
            mb_y = mb_y.to(device)

            outputs = model(mb_x, attention_mask=mb_m, labels=mb_y)
            loss = outputs[0]
            val_loss += loss.item() / len(dataloader)

    return val_loss


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def test(model, test_dataloader, device):
    outputs = []
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for k, (mb_x, mb_m, mb_y) in enumerate(test_dataloader):
            mb_x = mb_x.to(device)
            mb_m = mb_m.to(device)
            mb_y = mb_y.to(device)

            output = model(mb_x, attention_mask=mb_m)
            logits = output.logits
            _, predicted = torch.max(logits, dim=1)

            outputs.append(predicted.cpu())

            correct += (predicted == mb_y).sum().item()
            total += mb_y.size(0)

    accuracy = correct / total
    return outputs, accuracy
