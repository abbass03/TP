def is_transitive(R):
    n = len(R)
    for i in range(n):
        for j in range(n):
            if R[i][j]:
                for k in range(n):
                    if R[j][k] and not R[i][k]:
                        return False
    return True

def print_matrix(M):
    for row in M:
        print(" ".join(str(x) for x in row))

def main():
    R3 = [[1,1,1],
          [0,0,1],
          [0,0,1]]  

    not_transitive = [[0,1,0],
                      [0,0,1],
                      [0,0,0]]  

    identity = [[1,0,0],
                [0,1,0],
                [0,0,1]]

    tests = {
        "R3 (given)": R3,
        "Counterexample (not transitive)": not_transitive,
        "Identity (transitive)": identity,
    }

    for name, R in tests.items():
        print(f"\n{name}:")
        print_matrix(R)
        print("Transitive?", is_transitive(R))

if __name__ == "__main__":
    main()