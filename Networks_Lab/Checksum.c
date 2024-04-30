
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

int *decToBin(int dec, int n)
{
    int *bin = (int *)calloc(n, sizeof(int));
    for (int i = n - 1; i >= 0; i--)
    {
        bin[i] = dec % 2;
        dec /= 2;
    }
    return bin;
}

void checksumGen(int **data, int n, int k)
{
    int carrynum = 0;
    for (int j = k - 1; j >= 0; j--)
    {
        int sum = 0;
        for (int i = 0; i < n; i++)
        {
            sum += data[i][j];
        }
        sum += carrynum;
        data[n][j] = sum % 2;
        carrynum = sum > 0 ? sum >> 1 : 0;
    }
    int *carry = decToBin(carrynum, k);

    printf("-----------------\n");
    for (int i = 0; i < k; i++)
    {
        printf("%d ", data[n][i]);
        data[n][i] = data[n][i] == 0 ? 1 : 0;
    }
    printf("<-Sum\n");
    printf("-----------------\n");
    for (int i = 0; i < k; i++)
    {
        printf("%d ", data[n][i]);
    }
    printf("<-CHECKSUM\n");
    printf("-----------------\n");
}

void checksumChk(int **data, int n, int k)
{
    int *chkBucket = (int *)calloc(k, sizeof(int));
    int carrynum = 0;
    for (int j = k - 1; j >= 0; j--)
    {
        int sum = 0;
        for (int i = 0; i <= n; i++)
        {
            sum += data[i][j];
        }
        sum += carrynum;
        chkBucket[j] = sum % 2;
        carrynum = sum > 0 ? sum >> 1 : 0;
    }
    int *carry = decToBin(carrynum, k);

    printf("-----------------\n");
    bool accept = true;
    for (int i = 0; i < k; i++)
    {
        chkBucket[i] = chkBucket[i] == 0 ? 1 : 0;
        printf("%d ", chkBucket[i]);
        if (chkBucket[i] != 0)
            accept = false;
    }
    printf("<-CHECKSUM\n");
    printf("-----------------\n");
    printf("%s", (accept ? "Accepted!\n" : "Rejected!\n"));
}

int main()
{
    int n, k;
    printf("Sender Side ->\n");
    printf("Enter no of Segmets: ");
    scanf("%d", &n);
    printf("Enter bit lenght of each segmet: ");
    scanf("%d", &k);
    int len = (n + 1) * sizeof(int *) + (n + 1) * (k) * sizeof(int);
    int **data = (int **)malloc(len);
    int *ptr = (int *)(data + n + 1);
    for (int i = 0; i < n + 1; i++)
        data[i] = (ptr + k * i);
    for (int i = 0; i < n; i++)
    {
        printf("Enter segment[%d] (space separated): ", (i + 1));
        for (int j = 0; j < k; j++)
        {
            scanf("%d", &data[i][j]);
        }
    }
    checksumGen(data, n, k);

    printf("\nReceiver Side ->\n");
    printf("You are supposed to enter %d segments of %d length!\n", n, k);
    for (int i = 0; i < n; i++)
    {
        printf("Enter segment[%d] (space separated): ", (i + 1));
        for (int j = 0; j < k; j++)
        {
            scanf("%d", &data[i][j]);
        }
    }
    printf("Enter the checksum (space separated): ");
    for (int j = 0; j < k; j++)
        scanf("%d", &data[n][j]);

    checksumChk(data, n, k);
}
