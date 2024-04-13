#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void main()
{
  int max_point = 6;
  int a[50], sender_side[70], receiver_side[70], temp_storing[70];
  int t, i, j, k, string_len, n, hamming_code_length, sum = 0, pos = 0;

  // input data
  printf("Enter Length of Data String: ");
  scanf("%d", &string_len);
  printf("Enter Data String ->\n");
  for (i = 0; i < string_len; i++)
  {
    scanf("%d", &a[i]);
  }
  printf("-----------------------------------\n");

  for (i = 0, j = 0; i < string_len; i++)
  {
    for (k = 0; k < max_point; k++)
    {
      t = pow(2, k) - 1;
      if (j == t)
      {
        sender_side[j] = 0;
        j++;
      }
    }
    sender_side[j] = a[i];
    j++;
  }

  hamming_code_length = j;
  printf("Length of Hamming code: %d bits\n", hamming_code_length);
  n = hamming_code_length - string_len;
  printf("Number of Parity Bits: %d \n", n);

  int b[n];
  int m = n - 1;
  for (k = 0; k < n; k++)
  {
    t = pow(2, k) - 1;

    for (i = t; i < hamming_code_length;)
    {
      for (j = 0; j <= t; j++)
      {
        sum = sum + sender_side[i];
        i++;
        if (i >= hamming_code_length)
          break;
      }

      if (i >= hamming_code_length)
        break;

      for (j = 0; j <= t; j++)
      {
        i++;
        if (i >= hamming_code_length)
          break;
      }

      if (i >= hamming_code_length)
        break;
    }
    sender_side[t] = sum % 2;
    sum = 0;
    printf("P%d: %d\n", t + 1, sender_side[t]);
  }

  printf("\nHamming code: Sender side:   ");
  for (i = 0; i < hamming_code_length; i++)
  {
    printf("%d ", sender_side[i]);
  }

  printf("\nHamming code: Receiver side ->\n");
  for (i = 0; i < hamming_code_length; i++)
  {
    scanf("%d", &receiver_side[i]);
  }

  // Store data to compute further
  for (int i = 0; i < hamming_code_length; i++)
  {
    temp_storing[i] = receiver_side[i];
  }

  sum = 0;
  for (k = 0; k < n; k++)
  {
    t = pow(2, k) - 1;

    for (i = t; i < hamming_code_length;)
    {
      for (j = 0; j <= t; j++)
      {
        sum = sum + receiver_side[i];
        i++;
        if (i >= hamming_code_length)
          break;
      }

      if (i >= hamming_code_length)
        break;

      for (j = 0; j <= t; j++)
      {
        i++;
        if (i >= hamming_code_length)
          break;
      }

      if (i >= hamming_code_length)
        break;
    }
    b[m] = sum % 2;
    sum = 0;
    printf("P%d: %d\n", t + 1, b[m]);
    m--;
  }
  for (m = 0; m < n; m++)
  {
    pos = pos + b[n - m - 1] * pow(2, m);
  }

  int flag;
  for (flag = 0; flag < hamming_code_length; flag++)
    if (temp_storing[flag] != sender_side[flag])
    {
      break;
    }

  if (flag == hamming_code_length)
  {
    printf("No Error Detected!");
  }

  else
  {
    printf("Position of Error: %d\n", pos);
    if (receiver_side[pos - 1] == 0)
      receiver_side[pos - 1] = 1;
    else
      receiver_side[pos - 1] = 0;

    printf("\nHamming code: Receiver side: Error Corrected:  ");
    for (i = 0; i < hamming_code_length; i++)
    {
      printf("%d ", receiver_side[i]);
    }

    printf("\n-----------------------------------\n", string_len);
  }
}

// Output ->

// 1.

// Enter Length of Data String: 4
// Enter Data String ->
// 1
// 0
// 1
// 1
// -----------------------------------
// Length of Hamming code: 7 bits
// Number of Parity Bits: 3
// P1: 0
// P2: 1
// P4: 0

// Hamming code: Sender side:   0 1 1 0 0 1 1
// Hamming code: Receiver side ->
// 0
// 1
// 1
// 0
// 0
// 1
// 1
// P1: 0
// P2: 0
// P4: 0
// No Error Detected!

// 2.

// Enter Length of Data String: 4
// Enter Data String ->
// 1
// 0
// 1
// 1
// -----------------------------------
// Length of Hamming code: 7 bits
// Number of Parity Bits: 3
// P1: 0
// P2: 1
// P4: 0

// Hamming code: Sender side:   0 1 1 0 0 1 1
// Hamming code: Receiver side ->
// 0
// 1
// 1
// 1
// 0
// 1
// 1
// P1: 0
// P2: 0
// P4: 1
// Position of Error: 4

// Hamming code: Receiver side: Error Corrected:  0 1 1 0 0 1 1