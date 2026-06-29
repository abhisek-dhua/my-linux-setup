# Ultimate Guide to Data Structures and Algorithms (DSA) with Python

## Table of Contents

1. Introduction to DSA
2. Big O Notation & Complexity Analysis
3. Arrays & Lists
4. Stacks
5. Queues
6. Linked Lists
7. Trees
8. Binary Search Trees
9. Heaps & Priority Queues
10. Hash Tables (Dictionaries)
11. Graphs
12. Searching Algorithms
13. Sorting Algorithms
14. Recursion & Backtracking
15. Dynamic Programming
16. Greedy Algorithms
17. Divide and Conquer
18. Advanced Data Structures
19. Practice Problems & Solutions
20. Resources & Further Reading

---

## 1. Introduction to DSA

Data Structures and Algorithms (DSA) are the foundation of efficient programming and problem-solving. Mastering DSA is essential for technical interviews, competitive programming, and building scalable software.

- **Data Structure:** A way to organize and store data for efficient access and modification.
- **Algorithm:** A step-by-step procedure to solve a problem or perform a computation.

**Why DSA?**

- Efficient code (speed and memory)
- Solving complex problems
- Essential for interviews and real-world applications

This guide covers DSA concepts from basics to advanced, with clear explanations and Python code examples for each topic.

---

## 2. Big O Notation & Complexity Analysis

Big O notation describes the upper bound of an algorithm's running time or space requirements in terms of input size (n). It helps compare the efficiency of algorithms.

### Common Complexity Classes

- **O(1):** Constant time (independent of input size)
- **O(log n):** Logarithmic time (e.g., binary search)
- **O(n):** Linear time (e.g., traversing a list)
- **O(n log n):** Linearithmic time (e.g., merge sort)
- **O(n^2):** Quadratic time (e.g., bubble sort)
- **O(2^n):** Exponential time (e.g., recursive Fibonacci)

### Why Analyze Complexity?

- Predict performance for large inputs
- Choose the best algorithm for a problem
- Avoid inefficient code

### Time Complexity Examples

```python
# O(1) - Constant time
arr = [1, 2, 3, 4, 5]
def get_first_element(arr):
    return arr[0]

# O(n) - Linear time
def print_all_elements(arr):
    for x in arr:
        print(x)

# O(n^2) - Quadratic time
def print_all_pairs(arr):
    for i in arr:
        for j in arr:
            print(i, j)
```

### Space Complexity Example

```python
# O(n) space: creating a new list of size n
def copy_list(arr):
    new_list = arr[:]
    return new_list
```

### Best, Worst, and Average Case

- **Best Case:** Minimum time for any input
- **Worst Case:** Maximum time for any input
- **Average Case:** Expected time over all inputs

### Practical Tips

- Focus on the dominant term (ignore constants and lower-order terms)
- Use Big O to guide, but always test with real data

---

## 3. Arrays & Lists

Arrays and lists are the most basic data structures for storing collections of elements.

### Python Lists

- Dynamic arrays (can grow/shrink)
- Store elements of any type

```python
# Creating a list
arr = [1, 2, 3, 4, 5]

# Accessing elements
print(arr[0])  # 1

# Modifying elements
arr[2] = 10
print(arr)  # [1, 2, 10, 4, 5]

# Adding elements
arr.append(6)  # O(1) amortized
arr.insert(2, 7)  # O(n)

# Removing elements
arr.pop()  # O(1)
arr.remove(10)  # O(n)

# Slicing
print(arr[1:4])
```

### Complexity

- Access: O(1)
- Insert/Delete at end: O(1) amortized
- Insert/Delete at beginning/middle: O(n)
- Search: O(n)

### When to Use

- When you need fast access by index
- When order matters

---

## 4. Stacks

A stack is a Last-In-First-Out (LIFO) data structure. The last element added is the first to be removed.

### Operations

- **push(x):** Add x to the top
- **pop():** Remove and return the top element
- **peek():** Return the top element without removing
- **is_empty():** Check if stack is empty

### Python Implementation

```python
# Using list as stack
stack = []

# Push
stack.append(1)
stack.append(2)
stack.append(3)

# Pop
print(stack.pop())  # 3
print(stack)        # [1, 2]

# Peek
print(stack[-1])    # 2

# Check empty
print(len(stack) == 0)
```

### Complexity

- Push: O(1)
- Pop: O(1)
- Peek: O(1)

### Applications

- Undo functionality
- Expression evaluation (parentheses matching)
- Backtracking algorithms

---

## 5. Queues

A queue is a First-In-First-Out (FIFO) data structure. The first element added is the first to be removed.

### Operations

- **enqueue(x):** Add x to the end
- **dequeue():** Remove and return the front element
- **peek():** Return the front element without removing
- **is_empty():** Check if queue is empty

### Python Implementation

```python
from collections import deque

# Create queue
your_queue = deque()

# Enqueue
your_queue.append(1)
your_queue.append(2)
your_queue.append(3)

# Dequeue
print(your_queue.popleft())  # 1
print(your_queue)            # deque([2, 3])

# Peek
print(your_queue[0])         # 2

# Check empty
print(len(your_queue) == 0)
```

### Complexity

- Enqueue: O(1)
- Dequeue: O(1)
- Peek: O(1)

### Applications

- Scheduling tasks
- Breadth-First Search (BFS) in graphs
- Print/job queues

---

## 6. Linked Lists

A linked list is a linear data structure where elements (nodes) are stored in separate objects, each pointing to the next node.

### Types

- **Singly Linked List:** Each node points to the next
- **Doubly Linked List:** Each node points to both next and previous
- **Circular Linked List:** Last node points back to the first

### Node Structure

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
```

### Singly Linked List Example

```python
class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        last = self.head
        while last.next:
            last = last.next
        last.next = new_node

    def print_list(self):
        curr = self.head
        while curr:
            print(curr.data, end=' -> ')
            curr = curr.next
        print('None')

# Usage
ll = LinkedList()
ll.append(1)
ll.append(2)
ll.append(3)
ll.print_list()  # 1 -> 2 -> 3 -> None
```

### Complexity

- Access: O(n)
- Insert at head: O(1)
- Insert at tail: O(n)
- Delete: O(n)

### When to Use

- When frequent insertions/deletions are needed
- When memory usage is critical (no resizing needed)

---

## 7. Trees

A tree is a hierarchical data structure with a root node and child nodes, forming a parent-child relationship.

### Terminology

- **Root:** Top node
- **Leaf:** Node with no children
- **Height:** Longest path from root to leaf
- **Depth:** Distance from root to node
- **Subtree:** Tree formed by a node and its descendants

### Binary Tree Example

```python
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def inorder_traversal(root):
    if root:
        inorder_traversal(root.left)
        print(root.value, end=' ')
        inorder_traversal(root.right)

# Usage
root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)
inorder_traversal(root)  # 4 2 5 1 3
```

### Complexity

- Traversal: O(n)
- Insert/Delete/Search (general tree): O(n)

### Applications

- Hierarchical data (file systems)
- Expression parsing
- Routing algorithms

---

## 8. Binary Search Trees (BST)

A BST is a binary tree where each node's left child is less and right child is greater than the node itself.

### Properties

- Left subtree: values < node
- Right subtree: values > node
- No duplicate nodes (in standard BST)

### BST Operations

```python
class BSTNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

    def insert(self, value):
        if value < self.value:
            if self.left:
                self.left.insert(value)
            else:
                self.left = BSTNode(value)
        elif value > self.value:
            if self.right:
                self.right.insert(value)
            else:
                self.right = BSTNode(value)

    def search(self, value):
        if value == self.value:
            return True
        elif value < self.value and self.left:
            return self.left.search(value)
        elif value > self.value and self.right:
            return self.right.search(value)
        return False

# Usage
bst = BSTNode(10)
bst.insert(5)
bst.insert(15)
bst.insert(2)
print(bst.search(15))  # True
print(bst.search(7))   # False
```

### Complexity

- Search: O(h) (h = height, O(log n) for balanced BST)
- Insert: O(h)
- Delete: O(h)

### When to Use

- Fast search, insert, and delete (if tree is balanced)
- Dynamic sorted data

---

## 9. Heaps & Priority Queues

A heap is a specialized tree-based data structure that satisfies the heap property. In a max heap, the parent is always larger than its children; in a min heap, the parent is always smaller.

### Types

- **Max Heap:** Parent >= children
- **Min Heap:** Parent <= children

### Python Implementation (Min Heap)

```python
import heapq

# Using heapq module
heap = []
heapq.heappush(heap, 3)
heapq.heappush(heap, 1)
heapq.heappush(heap, 4)
heapq.heappush(heap, 2)

print(heapq.heappop(heap))  # 1 (smallest element)
print(heap)                 # [2, 3, 4]

# Heapify a list
arr = [3, 1, 4, 1, 5, 9, 2, 6]
heapq.heapify(arr)
print(arr)  # [1, 1, 2, 3, 5, 9, 4, 6]
```

### Custom Heap Implementation

```python
class MinHeap:
    def __init__(self):
        self.heap = []

    def parent(self, i):
        return (i - 1) // 2

    def left_child(self, i):
        return 2 * i + 1

    def right_child(self, i):
        return 2 * i + 2

    def swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    def insert(self, key):
        self.heap.append(key)
        self._bubble_up(len(self.heap) - 1)

    def _bubble_up(self, i):
        parent = self.parent(i)
        if i > 0 and self.heap[i] < self.heap[parent]:
            self.swap(i, parent)
            self._bubble_up(parent)

    def extract_min(self):
        if len(self.heap) == 0:
            return None
        if len(self.heap) == 1:
            return self.heap.pop()

        min_val = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._bubble_down(0)
        return min_val

    def _bubble_down(self, i):
        smallest = i
        left = self.left_child(i)
        right = self.right_child(i)

        if left < len(self.heap) and self.heap[left] < self.heap[smallest]:
            smallest = left
        if right < len(self.heap) and self.heap[right] < self.heap[smallest]:
            smallest = right

        if smallest != i:
            self.swap(i, smallest)
            self._bubble_down(smallest)

# Usage
heap = MinHeap()
heap.insert(3)
heap.insert(1)
heap.insert(4)
print(heap.extract_min())  # 1
```

### Complexity

- Insert: O(log n)
- Extract min/max: O(log n)
- Get min/max: O(1)
- Build heap: O(n)

### Applications

- Priority queues
- Heap sort
- Dijkstra's algorithm
- Top-k elements

---

## 10. Hash Tables (Dictionaries)

A hash table is a data structure that maps keys to values using a hash function to compute an index into an array of buckets.

### Python Dictionaries

```python
# Creating a dictionary
hash_table = {}

# Insert/Update
hash_table['apple'] = 1
hash_table['banana'] = 2
hash_table['cherry'] = 3

# Access
print(hash_table['apple'])  # 1

# Check if key exists
if 'apple' in hash_table:
    print("Key exists")

# Delete
del hash_table['banana']

# Iterate
for key, value in hash_table.items():
    print(f"{key}: {value}")
```

### Custom Hash Table Implementation

```python
class HashTable:
    def __init__(self, size=10):
        self.size = size
        self.table = [[] for _ in range(size)]

    def _hash(self, key):
        return hash(key) % self.size

    def insert(self, key, value):
        hash_key = self._hash(key)
        bucket = self.table[hash_key]

        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        bucket.append((key, value))

    def get(self, key):
        hash_key = self._hash(key)
        bucket = self.table[hash_key]

        for k, v in bucket:
            if k == key:
                return v
        return None

    def delete(self, key):
        hash_key = self._hash(key)
        bucket = self.table[hash_key]

        for i, (k, v) in enumerate(bucket):
            if k == key:
                del bucket[i]
                return

# Usage
ht = HashTable()
ht.insert('apple', 1)
ht.insert('banana', 2)
print(ht.get('apple'))  # 1
ht.delete('apple')
print(ht.get('apple'))  # None
```

### Complexity

- Average case: O(1) for insert, delete, search
- Worst case: O(n) if many collisions

### Applications

- Database indexing
- Caching
- Symbol tables in compilers
- Removing duplicates

---

## 11. Graphs

A graph is a collection of vertices (nodes) connected by edges. Graphs can be directed or undirected, weighted or unweighted.

### Types

- **Undirected:** Edges have no direction
- **Directed:** Edges have direction
- **Weighted:** Edges have weights
- **Unweighted:** Edges have no weights

### Adjacency List Representation

```python
from collections import defaultdict

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, u, v):
        self.graph[u].append(v)

    def print_graph(self):
        for vertex in self.graph:
            print(f"{vertex} -> {self.graph[vertex]}")

# Usage
g = Graph()
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 2)
g.add_edge(2, 0)
g.add_edge(2, 3)
g.add_edge(3, 3)
g.print_graph()
```

### Adjacency Matrix Representation

```python
class GraphMatrix:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for _ in range(vertices)] for _ in range(vertices)]

    def add_edge(self, u, v):
        self.graph[u][v] = 1
        self.graph[v][u] = 1  # For undirected graph

    def print_graph(self):
        for row in self.graph:
            print(row)

# Usage
g = GraphMatrix(4)
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 2)
g.add_edge(2, 3)
g.print_graph()
```

### Graph Traversal

```python
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    print(start, end=' ')

    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

def bfs(graph, start):
    visited = set()
    queue = [start]
    visited.add(start)

    while queue:
        vertex = queue.pop(0)
        print(vertex, end=' ')

        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

# Usage
graph = {
    0: [1, 2],
    1: [2],
    2: [0, 3],
    3: [3]
}
print("DFS:", end=' ')
dfs(graph, 2)
print("\nBFS:", end=' ')
bfs(graph, 2)
```

### Complexity

- Space: O(V + E) for adjacency list, O(V²) for adjacency matrix
- DFS/BFS: O(V + E)
- Add edge: O(1) for adjacency list, O(1) for adjacency matrix

### Applications

- Social networks
- Navigation systems
- Network routing
- Dependency resolution

---

## 12. Searching Algorithms

Searching algorithms find the location of a target value within a data structure.

### Linear Search

Searches sequentially through each element until the target is found.

```python
def linear_search(arr, target):
    for i, element in enumerate(arr):
        if element == target:
            return i
    return -1

# Usage
arr = [64, 34, 25, 12, 22, 11, 90]
print(linear_search(arr, 22))  # 4
print(linear_search(arr, 100))  # -1
```

**Complexity:** O(n)

### Binary Search

Searches in a sorted array by repeatedly dividing the search interval in half.

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# Usage
arr = [11, 12, 22, 25, 34, 64, 90]
print(binary_search(arr, 22))  # 2
print(binary_search(arr, 100))  # -1
```

**Complexity:** O(log n)

### Recursive Binary Search

```python
def binary_search_recursive(arr, target, left, right):
    if left > right:
        return -1

    mid = (left + right) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)

# Usage
arr = [11, 12, 22, 25, 34, 64, 90]
print(binary_search_recursive(arr, 22, 0, len(arr) - 1))  # 2
```

---

## 13. Sorting Algorithms

Sorting algorithms arrange elements in a specific order (ascending or descending).

### Bubble Sort

Repeatedly steps through the list, compares adjacent elements, and swaps them if they are in the wrong order.

```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

# Usage
arr = [64, 34, 25, 12, 22, 11, 90]
print(bubble_sort(arr.copy()))  # [11, 12, 22, 25, 34, 64, 90]
```

**Complexity:** O(n²)

### Selection Sort

Divides the input list into a sorted and unsorted region, repeatedly selects the smallest element from the unsorted region.

```python
def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

# Usage
arr = [64, 34, 25, 12, 22, 11, 90]
print(selection_sort(arr.copy()))  # [11, 12, 22, 25, 34, 64, 90]
```

**Complexity:** O(n²)

### Insertion Sort

Builds the final sorted array one item at a time by repeatedly inserting a new element into the sorted portion.

```python
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

# Usage
arr = [64, 34, 25, 12, 22, 11, 90]
print(insertion_sort(arr.copy()))  # [11, 12, 22, 25, 34, 64, 90]
```

**Complexity:** O(n²)

### Merge Sort

A divide-and-conquer algorithm that recursively breaks down a problem into smaller subproblems.

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Usage
arr = [64, 34, 25, 12, 22, 11, 90]
print(merge_sort(arr))  # [11, 12, 22, 25, 34, 64, 90]
```

**Complexity:** O(n log n)

### Quick Sort

A divide-and-conquer algorithm that picks a 'pivot' element and partitions the array around it.

```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quick_sort(left) + middle + quick_sort(right)

# Usage
arr = [64, 34, 25, 12, 22, 11, 90]
print(quick_sort(arr))  # [11, 12, 22, 25, 34, 64, 90]
```

**Complexity:** O(n log n) average, O(n²) worst case

---

## 14. Recursion & Backtracking

Recursion is a method where the solution to a problem depends on solutions to smaller instances of the same problem.

### Basic Recursion Examples

#### Factorial

```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

# Usage
print(factorial(5))  # 120
```

#### Fibonacci

```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Usage
print(fibonacci(10))  # 55
```

#### Tower of Hanoi

```python
def tower_of_hanoi(n, source, auxiliary, target):
    if n == 1:
        print(f"Move disk 1 from {source} to {target}")
        return

    tower_of_hanoi(n - 1, source, target, auxiliary)
    print(f"Move disk {n} from {source} to {target}")
    tower_of_hanoi(n - 1, auxiliary, source, target)

# Usage
tower_of_hanoi(3, 'A', 'B', 'C')
```

### Backtracking

Backtracking is a technique for finding all (or some) solutions to computational problems, particularly constraint satisfaction problems.

#### N-Queens Problem

```python
def solve_n_queens(n):
    def is_safe(board, row, col):
        for i in range(row):
            if board[i][col] == 'Q':
                return False

        for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
            if board[i][j] == 'Q':
                return False

        for i, j in zip(range(row, -1, -1), range(col, n)):
            if board[i][j] == 'Q':
                return False

        return True

    def solve(board, row):
        if row == n:
            return True

        for col in range(n):
            if is_safe(board, row, col):
                board[row][col] = 'Q'
                if solve(board, row + 1):
                    return True
                board[row][col] = '.'

        return False

    board = [['.' for _ in range(n)] for _ in range(n)]
    if solve(board, 0):
        return board
    return None

# Usage
solution = solve_n_queens(4)
for row in solution:
    print(row)
```

#### Subset Sum

```python
def subset_sum(arr, target):
    def backtrack(index, current_sum, path):
        if current_sum == target:
            result.append(path[:])
            return

        if current_sum > target or index >= len(arr):
            return

        path.append(arr[index])
        backtrack(index + 1, current_sum + arr[index], path)
        path.pop()
        backtrack(index + 1, current_sum, path)

    result = []
    backtrack(0, 0, [])
    return result

# Usage
arr = [2, 4, 6, 8]
target = 10
print(subset_sum(arr, target))  # [[2, 4, 4], [2, 8], [4, 6]]
```

### Recursion vs Iteration

- **Recursion:** More elegant, easier to understand for some problems
- **Iteration:** More efficient in terms of space complexity

### Tips for Recursion

1. Always have a base case
2. Ensure the recursive case moves toward the base case
3. Be aware of stack overflow for deep recursion

---

## 15. Dynamic Programming

Dynamic Programming (DP) is a method for solving complex problems by breaking them down into simpler subproblems. It stores the results of subproblems to avoid redundant calculations.

### Key Concepts

- **Optimal Substructure:** Optimal solution contains optimal solutions to subproblems
- **Overlapping Subproblems:** Same subproblems are solved multiple times
- **Memoization:** Store results of expensive function calls

### Fibonacci with Memoization

```python
def fibonacci_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n

    memo[n] = fibonacci_memo(n - 1, memo) + fibonacci_memo(n - 2, memo)
    return memo[n]

# Usage
print(fibonacci_memo(50))  # Fast calculation
```

### Longest Common Subsequence (LCS)

```python
def lcs(str1, str2):
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]

# Usage
str1 = "ABCDGH"
str2 = "AEDFHR"
print(lcs(str1, str2))  # 3 (ADH)
```

### Knapsack Problem

```python
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
            else:
                dp[i][w] = dp[i - 1][w]

    return dp[n][capacity]

# Usage
weights = [1, 3, 4, 5]
values = [1, 4, 5, 7]
capacity = 7
print(knapsack(weights, values, capacity))  # 9
```

### Coin Change

```python
def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)

    return dp[amount] if dp[amount] != float('inf') else -1

# Usage
coins = [1, 2, 5]
amount = 11
print(coin_change(coins, amount))  # 3
```

---

## 16. Greedy Algorithms

Greedy algorithms make locally optimal choices at each step with the hope of finding a global optimum.

### Activity Selection Problem

```python
def activity_selection(start, finish):
    n = len(start)
    selected = [0]  # First activity is always selected
    j = 0

    for i in range(1, n):
        if start[i] >= finish[j]:
            selected.append(i)
            j = i

    return selected

# Usage
start = [1, 3, 0, 5, 8, 5]
finish = [2, 4, 6, 7, 9, 9]
print(activity_selection(start, finish))  # [0, 1, 3, 4]
```

### Huffman Coding

```python
import heapq

class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(characters, frequencies):
    heap = [HuffmanNode(char, freq) for char, freq in zip(characters, frequencies)]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)

        internal = HuffmanNode(None, left.freq + right.freq)
        internal.left = left
        internal.right = right

        heapq.heappush(heap, internal)

    return heap[0]

# Usage
characters = ['a', 'b', 'c', 'd', 'e', 'f']
frequencies = [5, 9, 12, 13, 16, 45]
root = build_huffman_tree(characters, frequencies)
```

### Dijkstra's Algorithm

```python
import heapq

def dijkstra(graph, start):
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] = 0
    pq = [(0, start)]

    while pq:
        current_distance, current_vertex = heapq.heappop(pq)

        if current_distance > distances[current_vertex]:
            continue

        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))

    return distances

# Usage
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'C': 2, 'D': 5},
    'C': {'A': 4, 'B': 2, 'D': 1},
    'D': {'B': 5, 'C': 1}
}
print(dijkstra(graph, 'A'))  # {'A': 0, 'B': 1, 'C': 3, 'D': 4}
```

---

## 17. Divide and Conquer

Divide and Conquer is an algorithmic paradigm that recursively breaks down a problem into smaller subproblems until they become simple enough to solve directly.

### Merge Sort (Revisited)

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result
```

### Quick Sort (Revisited)

```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quick_sort(left) + middle + quick_sort(right)
```

### Strassen's Matrix Multiplication

```python
def strassen_multiply(A, B):
    n = len(A)

    if n == 1:
        return [[A[0][0] * B[0][0]]]

    # Divide matrices into quadrants
    mid = n // 2
    A11 = [row[:mid] for row in A[:mid]]
    A12 = [row[mid:] for row in A[:mid]]
    A21 = [row[:mid] for row in A[mid:]]
    A22 = [row[mid:] for row in A[mid:]]

    B11 = [row[:mid] for row in B[:mid]]
    B12 = [row[mid:] for row in B[:mid]]
    B21 = [row[:mid] for row in B[mid:]]
    B22 = [row[mid:] for row in B[mid:]]

    # Strassen's formulas
    P1 = strassen_multiply(A11, matrix_subtract(B12, B22))
    P2 = strassen_multiply(matrix_add(A11, A12), B22)
    P3 = strassen_multiply(matrix_add(A21, A22), B11)
    P4 = strassen_multiply(A22, matrix_subtract(B21, B11))
    P5 = strassen_multiply(matrix_add(A11, A22), matrix_add(B11, B22))
    P6 = strassen_multiply(matrix_subtract(A12, A22), matrix_add(B21, B22))
    P7 = strassen_multiply(matrix_subtract(A11, A21), matrix_add(B11, B12))

    # Combine results
    C11 = matrix_add(matrix_subtract(matrix_add(P5, P4), P2), P6)
    C12 = matrix_add(P1, P2)
    C21 = matrix_add(P3, P4)
    C22 = matrix_subtract(matrix_subtract(matrix_add(P5, P1), P3), P7)

    # Combine quadrants
    result = []
    for i in range(mid):
        result.append(C11[i] + C12[i])
    for i in range(mid):
        result.append(C21[i] + C22[i])

    return result

def matrix_add(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def matrix_subtract(A, B):
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
```

### Closest Pair of Points

```python
import math

def closest_pair(points):
    def distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def closest_pair_recursive(sorted_x, sorted_y):
        n = len(sorted_x)

        if n <= 3:
            min_dist = float('inf')
            for i in range(n):
                for j in range(i + 1, n):
                    dist = distance(sorted_x[i], sorted_x[j])
                    if dist < min_dist:
                        min_dist = dist
            return min_dist

        mid = n // 2
        mid_x = sorted_x[mid][0]

        left_x = sorted_x[:mid]
        right_x = sorted_x[mid:]
        left_y = [p for p in sorted_y if p[0] <= mid_x]
        right_y = [p for p in sorted_y if p[0] > mid_x]

        left_dist = closest_pair_recursive(left_x, left_y)
        right_dist = closest_pair_recursive(right_x, right_y)
        min_dist = min(left_dist, right_dist)

        # Check strip
        strip = [p for p in sorted_y if abs(p[0] - mid_x) < min_dist]
        for i in range(len(strip)):
            for j in range(i + 1, min(i + 7, len(strip))):
                dist = distance(strip[i], strip[j])
                if dist < min_dist:
                    min_dist = dist

        return min_dist

    sorted_x = sorted(points, key=lambda p: p[0])
    sorted_y = sorted(points, key=lambda p: p[1])
    return closest_pair_recursive(sorted_x, sorted_y)

# Usage
points = [(2, 3), (12, 30), (40, 50), (5, 1), (12, 10), (3, 4)]
print(closest_pair(points))
```

---

## 18. Advanced Data Structures

### Trie (Prefix Tree)

A tree data structure used to store a dynamic set of strings, where each node represents a character.

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

# Usage
trie = Trie()
trie.insert("apple")
trie.insert("app")
print(trie.search("apple"))  # True
print(trie.starts_with("app"))  # True
```

### Segment Tree

A tree data structure for storing information about intervals, allowing efficient range queries.

```python
class SegmentTree:
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self.build(arr, 0, 0, self.n - 1)

    def build(self, arr, node, start, end):
        if start == end:
            self.tree[node] = arr[start]
            return

        mid = (start + end) // 2
        self.build(arr, 2 * node + 1, start, mid)
        self.build(arr, 2 * node + 2, mid + 1, end)
        self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]

    def query(self, node, start, end, l, r):
        if r < start or l > end:
            return 0
        if l <= start and r >= end:
            return self.tree[node]

        mid = (start + end) // 2
        left = self.query(2 * node + 1, start, mid, l, r)
        right = self.query(2 * node + 2, mid + 1, end, l, r)
        return left + right

    def update(self, node, start, end, index, value):
        if start == end:
            self.tree[node] = value
            return

        mid = (start + end) // 2
        if index <= mid:
            self.update(2 * node + 1, start, mid, index, value)
        else:
            self.update(2 * node + 2, mid + 1, end, index, value)
        self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]

# Usage
arr = [1, 3, 5, 7, 9, 11]
st = SegmentTree(arr)
print(st.query(0, 0, len(arr) - 1, 1, 3))  # 15 (3 + 5 + 7)
```

### Disjoint Set (Union-Find)

A data structure that tracks a set of elements partitioned into a number of disjoint subsets.

```python
class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return

        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

# Usage
ds = DisjointSet(5)
ds.union(0, 1)
ds.union(2, 3)
ds.union(1, 2)
print(ds.find(0) == ds.find(3))  # True
```

### Skip List

A probabilistic data structure that allows fast search within an ordered sequence of elements.

```python
import random

class SkipListNode:
    def __init__(self, value, level):
        self.value = value
        self.forward = [None] * (level + 1)

class SkipList:
    def __init__(self, max_level=16):
        self.max_level = max_level
        self.level = 0
        self.header = SkipListNode(None, max_level)

    def random_level(self):
        level = 0
        while random.random() < 0.5 and level < self.max_level:
            level += 1
        return level

    def insert(self, value):
        update = [None] * (self.max_level + 1)
        current = self.header

        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].value < value:
                current = current.forward[i]
            update[i] = current

        current = current.forward[0]

        if current is None or current.value != value:
            new_level = self.random_level()

            if new_level > self.level:
                for i in range(self.level + 1, new_level + 1):
                    update[i] = self.header
                self.level = new_level

            new_node = SkipListNode(value, new_level)

            for i in range(new_level + 1):
                new_node.forward[i] = update[i].forward[i]
                update[i].forward[i] = new_node

    def search(self, value):
        current = self.header

        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].value < value:
                current = current.forward[i]

        current = current.forward[0]

        if current and current.value == value:
            return True
        return False

# Usage
sl = SkipList()
sl.insert(3)
sl.insert(6)
sl.insert(7)
sl.insert(9)
print(sl.search(6))  # True
print(sl.search(5))  # False
```

---

## 19. Practice Problems & Solutions

### Problem 1: Two Sum

Given an array of integers and a target sum, return indices of two numbers that add up to the target.

```python
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

# Usage
nums = [2, 7, 11, 15]
target = 9
print(two_sum(nums, target))  # [0, 1]
```

### Problem 2: Valid Parentheses

Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

```python
def is_valid_parentheses(s):
    stack = []
    brackets = {')': '(', '}': '{', ']': '['}

    for char in s:
        if char in '({[':
            stack.append(char)
        elif char in ')}]':
            if not stack or stack.pop() != brackets[char]:
                return False

    return len(stack) == 0

# Usage
print(is_valid_parentheses("()[]{}"))  # True
print(is_valid_parentheses("([)]"))    # False
```

### Problem 3: Maximum Subarray

Find the contiguous subarray with the largest sum.

```python
def max_subarray(nums):
    max_sum = current_sum = nums[0]

    for num in nums[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)

    return max_sum

# Usage
nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
print(max_subarray(nums))  # 6
```

### Problem 4: Merge Two Sorted Lists

Merge two sorted linked lists and return it as a new sorted list.

```python
def merge_sorted_lists(l1, l2):
    dummy = ListNode(0)
    current = dummy

    while l1 and l2:
        if l1.val <= l2.val:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next

    current.next = l1 if l1 else l2
    return dummy.next
```

### Problem 5: Binary Tree Level Order Traversal

Return the level order traversal of a binary tree's nodes' values.

```python
from collections import deque

def level_order_traversal(root):
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level = []
        level_size = len(queue)

        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(level)

    return result
```

### Problem 6: LRU Cache

Design and implement a data structure for Least Recently Used (LRU) cache.

```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return -1

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value

        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

# Usage
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
print(cache.get(1))  # 1
cache.put(3, 3)
print(cache.get(2))  # -1
```

---

## 20. Resources & Further Reading

### Online Platforms

1. **LeetCode**: Practice coding problems
2. **HackerRank**: Algorithm challenges
3. **Codeforces**: Competitive programming
4. **GeeksforGeeks**: DSA tutorials and problems
5. **InterviewBit**: Interview preparation

### Books

1. **"Introduction to Algorithms"** by Cormen, Leiserson, Rivest, Stein
2. **"Algorithm Design Manual"** by Steven Skiena
3. **"Programming Pearls"** by Jon Bentley
4. **"Cracking the Coding Interview"** by Gayle McDowell
5. **"Elements of Programming Interviews"** by Adnan Aziz

### Courses

1. **MIT 6.006**: Introduction to Algorithms
2. **Stanford CS161**: Design and Analysis of Algorithms
3. **Princeton COS226**: Data Structures and Algorithms
4. **UC Berkeley CS61B**: Data Structures

### YouTube Channels

1. **Back To Back SWE**: Algorithm explanations
2. **Tushar Roy**: Dynamic programming tutorials
3. **Abdul Bari**: Algorithm concepts
4. **mycodeschool**: Data structure tutorials

### Practice Problems by Category

#### Arrays & Strings

- Two Sum
- Valid Parentheses
- Maximum Subarray
- Longest Substring Without Repeating Characters
- Container With Most Water

#### Linked Lists

- Reverse Linked List
- Detect Cycle
- Merge Two Sorted Lists
- Remove Nth Node From End
- Add Two Numbers

#### Trees

- Binary Tree Inorder Traversal
- Maximum Depth of Binary Tree
- Validate Binary Search Tree
- Lowest Common Ancestor
- Serialize and Deserialize Binary Tree

#### Graphs

- Number of Islands
- Course Schedule
- Clone Graph
- Word Ladder
- Minimum Height Trees

#### Dynamic Programming

- Climbing Stairs
- House Robber
- Longest Increasing Subsequence
- Edit Distance
- Regular Expression Matching

### Interview Tips

1. **Understand the problem**: Ask clarifying questions
2. **Think out loud**: Explain your thought process
3. **Start with brute force**: Then optimize
4. **Consider edge cases**: Empty inputs, single elements, etc.
5. **Test your code**: Walk through examples
6. **Discuss complexity**: Time and space complexity

### Common Mistakes to Avoid

1. Not handling edge cases
2. Ignoring space complexity
3. Not considering multiple solutions
4. Rushing to code without planning
5. Not testing with examples

---

## Conclusion

This comprehensive guide covers the fundamental concepts of Data Structures and Algorithms with Python. Remember:

1. **Practice regularly**: Solve problems daily
2. **Understand concepts**: Don't just memorize solutions
3. **Analyze complexity**: Always consider time and space complexity
4. **Learn from mistakes**: Review and understand wrong solutions
5. **Stay consistent**: Regular practice is key to mastery

The field of DSA is vast and constantly evolving. This guide provides a solid foundation, but continuous learning and practice are essential for mastery. Start with the basics, build projects, and gradually explore more advanced topics based on your interests and goals.

Happy coding and problem-solving! 🚀
