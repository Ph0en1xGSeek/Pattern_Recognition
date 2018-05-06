#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

struct node
{
    int label;
    double score;
    double precision;
    double recall;
    double AUC_PR;
    double AP;
    bool operator<(node b) const
    {
        return score > b.score;
    }
};

int main()
{
    vector<node> arr;
    const double init_p = 1.0;
    const double init_r = 0.0;
    int label;
    double score;
    int cnt = 0;
    int Pcnt = 0;
    int Ncnt = 0;
    while(~scanf("%d %lf", &label, &score))
    {
        node tmp;
        tmp.label = label;
        tmp.score = score;
        arr.push_back(tmp);
        cnt++;
        if(label == 1) Pcnt++;
        else if(label == 2) Ncnt++;
    }
    sort(arr.begin(), arr.end());
    int curPcnt = 0;
    double AUC_PR_sum = 0.0;
    double AP_sum = 0.0;
    for(int i = 0; i < cnt; i++)
    {
        if(arr[i].label == 1) curPcnt++;
        arr[i].precision = curPcnt * 1.0 / (i+1);
        arr[i].recall = curPcnt * 1.0 / (Pcnt);
        if(i == 0)
        {
            arr[i].AUC_PR = (arr[i].recall - init_r) * (arr[i].precision + init_p) / 2.0;
            arr[i].AP = (arr[i].recall - init_r) * arr[i].precision;
        }
        else
        {
            arr[i].AUC_PR = (arr[i].recall - arr[i-1].recall) * (arr[i].precision + arr[i-1].precision) / 2.0;
            arr[i].AP = (arr[i].recall - arr[i-1].recall) * arr[i].precision;
        }
        AUC_PR_sum += arr[i].AUC_PR;
        AP_sum += arr[i].AP;
    }
    for(int i = 0; i < cnt; i++)
    {
        printf("%-3d%-3d%-8.1f%-8.4f%-8.4f%-8.4f%-8.4f\n", i+1, arr[i].label,
               arr[i].score, arr[i].precision, arr[i].recall,
               arr[i].AUC_PR, arr[i].AP);
    }
    printf("Sum: %8.4f%8.4f\n", AUC_PR_sum, AP_sum);
    return 0;
}
