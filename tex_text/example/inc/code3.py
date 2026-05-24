opt = torch.optim.AdamW(model.parameters(),
                        lr=1e-4, weight_decay=5e-4)

for epoch in range(50):
    lr_mult = cosine_warmup(epoch, total=50, warmup=3)
    for pg in opt.param_groups:
        pg['lr'] = pg['_base_lr'] * lr_mult

    model.train()
    for imgs, bboxes, labels in train_loader:
        imgs, bboxes = imgs.to(device), bboxes.to(device)
        opt.zero_grad()
        logits = model(imgs, bboxes)
        total = 0.0
        for name in criteria:
            tg = labels[name].to(device)
            total = total + loss_fns[name](logits[safe[name]], tg)
        loss = total / len(criteria)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    f1 = validate(model, val_loader)
    if f1 > best_f1:
        torch.save(model.state_dict(), 'best.pth')
        best_f1 = f1
