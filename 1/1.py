def main():
    n = input()
    try:
        n = int(n)
        if n <= 0:
            print("Given number is NOT positive integer.")
            return
    except ValueError:
        print("Given input is NOT a number.")
        return
    result = (n * (n + 1)) // 2
    print(result)
    return


if __name__ == '__main__':
    main()
