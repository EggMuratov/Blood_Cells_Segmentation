def test_model(model, test_loader):
    model.eval()
    total_iou = 0.0

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            total_iou += iou_score(outputs, masks.detach())

    return total_iou / len(test_loader)


def train_model(train_loader, test_loader, num_epochs):
    train_losses = []
    train_ious = []
    test_ious = [0.6]
    best_score = 0.6
    lr = 0.001

    for epoch in range(num_epochs):
        model = torch.load('best_Blood_Cells_model_unet_version_5.pth', map_location=device)
        if len(test_ious) >= 2:
            if test_ious[-1] < test_ious[-2]:
                lr *= 0.1
        optimizer = optim.Adam(params=model.parameters(), lr=lr)
        model.train()
        epoch_loss = 0.0
        epoch_iou = 0.0

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
            start_time = time.time()
            i = 0
            for images, masks in train_loader:
                i += 1
                images = images.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()

                outputs = model(images)

                loss = dice_loss(outputs, masks)
                epoch_loss += loss.item()

                loss.backward()
                optimizer.step()

                epoch_iou += iou_score(outputs, masks)

                pbar.set_postfix({"Loss": loss.item(), "IoU": epoch_iou / (pbar.n + 1)})
                pbar.update(1)

            epoch_loss /= len(train_loader)
            epoch_iou /= len(train_loader)
            train_losses.append(epoch_loss)
            train_ious.append(epoch_iou)

            elapsed_time = time.time() - start_time
            estimated_time = elapsed_time * (num_epochs - (epoch + 1))
            print(f'\nEpoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, IoU: {epoch_iou:.4f}, \n'
                  f'Elapsed Time: {int(elapsed_time)}s, Estimated Time Left: {int(estimated_time)}s, Learning Rate: {lr}\n')

            test_iou = test_model(model, test_loader)
            test_ious.append(test_iou)
            print(f'Test IoU: {test_iou:.4f}')
            if best_score < test_iou:
                best_score = test_iou
                torch.save(model, 'best_Blood_Cells_model_unet_version_5.pth')
                print(f"MODEL WITH SCORE {best_score} SAVED!!!")
