#include <stdio.h>
#include <stdbool.h>
#include <unistd.h>

int main()
{
  int totalFrames;
  printf("Enter total frames to send.\n");
  scanf("%d", &totalFrames);

  int frameToSend = 1;
  int ack;

  while (frameToSend <= totalFrames)
  {
    printf("Sending frame %d\n", frameToSend);
    printf("Enter acknowledgment for frame %d (1 for received): ", frameToSend);
    scanf("%d", &ack);
    if (ack == 0)
    {
      printf("\nWaiting for %d seconds\n", 1);
      sleep(1);
    }
    if (ack == 1)
    {
      printf("Receiver: Received all %d frames \n", frameToSend);
      frameToSend++;
    }
  }

  return 0;
}

// output -

// Enter total frames to send.
// 4
// Sending frame 1
// Enter acknowledgment for frame 1 (1 for received): 1
// Receiver: Received frame 1
// Sending frame 2
// Enter acknowledgment for frame 2 (1 for received): 0

// Waiting for 1 seconds
// Sending frame 2
// Enter acknowledgment for frame 2 (1 for received): 1
// Receiver: Received frame 2
// Sending frame 3
// Enter acknowledgment for frame 3 (1 for received): 0

// Waiting for 1 seconds
// Sending frame 3
// Enter acknowledgment for frame 3 (1 for received): 1
// Receiver: Received frame 3
// Sending frame 4
// Enter acknowledgment for frame 4 (1 for received): 1
// Receiver: Received frame 4
