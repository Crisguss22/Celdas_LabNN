from labrynth_nn import LabrynthNN
import time

if __name__ == "__main__":
    print(time.strftime('%X %x'))
    LabrynthNN().train()
    print(time.strftime('%X %x'))
        
