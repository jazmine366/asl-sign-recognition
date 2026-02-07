# ASL Sign Recognition System 🖐

Real time American Sign Language (ASL) recognition using MediaPipe hand landmarks and a PyTorch LSTM model. 
User signs and computer says the signs utilizing Text to Speech feature.

## Current Labels
**Letters:** A, B, M, N  
**Words:** Hi, My, Name

## Current Status
- Real time recognition pipeline implemented
- Stable recognition for letters and head based words
- Trade off identified between gesture accuracy and latency

## Known Issues
- **"My"** is not fully solved confused with letter sign "A"
- Chest based signs causes higher latency 
- Hi has 2 versions in ASL, both versions were collected at first yet, Hi (signed B letter downward) is confused between B and Hi. Therefore only one version of **Hi** is kept.

## Next Steps
- Add pose landmarks for body relative normalization  (solve My)
- Reduce latency while maintaining stability

## Varieties 
- different lighting
- closer/farther distance to camera
- left and right hand signing collected


> Project is under development.
