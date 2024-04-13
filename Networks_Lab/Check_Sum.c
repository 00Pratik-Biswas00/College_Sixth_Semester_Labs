#include <stdio.h>
#include <stdlib.h>

int sender(int arr[10], int n)
{
  int check_sum, sum = 0, i;
  for (int i = 0; i < n; i++)
    sum += arr[i];
  printf("\nSum is: %d", sum);
  check_sum = ~sum;
  printf("\nChecksum is: %d", check_sum);
  return check_sum;
}

void receiver(int arr[10], int sch)
{
  int check_sum, sum = 0, i;
  printf("\n\nReceiver Side");
  printf("\nEnter size of the string for the receiver side: ");
  int n;
  scanf("%d", &n);

  printf("Enter elements of the array for the receiver side ->\n");
  for (int i = 0; i < n; i++)
    scanf("%d", &arr[i]);
  for (int i = 0; i < n; i++)
    sum += arr[i];
  printf("Sum is: %d", sum);
  sum = sum + sch;
  check_sum = ~sum;
  printf("\nCheck Sum is: %d", check_sum);
}

int main()
{
  int n, sch, rch;
  printf("\n\nSender Side");
  printf("\nEnter size of the string for the Sender side: ");
  scanf("%d", &n);
  int arr[n];
  printf("Enter elements of the array for the receiver side ->\n");
  for (int i = 0; i < n; i++)
    scanf("%d", &arr[i]);
  sch = sender(arr, n);
  receiver(arr, sch);
}