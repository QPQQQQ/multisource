import sys

if __name__ == "__main__":
    n = int(sys.stdin.readline().strip())
    list1 = sys.stdin.readline().strip()
    list2 = sys.stdin.readline().strip()
    list1 = list(map(int, list1.split()))
    list2 = list(map(int, list2.split()))
    list3 = sorted(list1)
    list4 = sorted(list2)[::-1]
    ans = 0
    for i in range(n):
        ans += max(0, list3[i] - list4[i])
        