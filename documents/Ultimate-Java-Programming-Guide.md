# Ultimate Java Programming Guide

## Table of Contents

1. [Introduction to Java](#introduction-to-java)
2. [Getting Started](#getting-started)
3. [Basic Syntax](#basic-syntax)
4. [Data Types](#data-types)
5. [Control Flow](#control-flow)
6. [Methods](#methods)
7. [Object-Oriented Programming](#object-oriented-programming)
8. [Packages and Imports](#packages-and-imports)
9. [Exception Handling](#exception-handling)
10. [File I/O](#file-io)
11. [Collections Framework](#collections-framework)
12. [Generics](#generics)
13. [Multithreading](#multithreading)
14. [Lambda Expressions and Streams](#lambda-expressions-and-streams)
15. [Annotations](#annotations)
16. [Reflection](#reflection)
17. [Serialization](#serialization)
18. [Networking](#networking)
19. [Database Connectivity](#database-connectivity)
20. [Best Practices](#best-practices)
21. [Common Libraries](#common-libraries)
22. [Debugging and Testing](#debugging-and-testing)
23. [Performance Optimization](#performance-optimization)
24. [Java Memory Management](#java-memory-management)
25. [Design Patterns](#design-patterns)

---

## Introduction to Java

Java is a high-level, class-based, object-oriented programming language designed to have as few implementation dependencies as possible. Created by James Gosling at Sun Microsystems in 1995, Java is known for its portability, robustness, and security.

### History and Evolution

- **1995**: Java 1.0 released by Sun Microsystems
- **2006**: Java becomes open source (OpenJDK)
- **2010**: Oracle acquires Sun Microsystems
- **2014**: Java 8 introduces Lambda expressions and Streams
- **2017**: Java 9 introduces modules
- **2018**: Java 11 becomes LTS (Long Term Support)
- **2021**: Java 17 becomes LTS
- **2023**: Java 21 becomes LTS

### Key Features

- **Platform Independent**: Write Once, Run Anywhere (WORA) - Java bytecode runs on any JVM
- **Object-Oriented**: Everything is an object with encapsulation, inheritance, and polymorphism
- **Strongly Typed**: Type checking at compile time prevents many runtime errors
- **Automatic Memory Management**: Garbage collection handles memory allocation and deallocation
- **Rich Standard Library**: Extensive APIs for networking, I/O, data structures, GUI, etc.
- **Multithreaded**: Built-in support for concurrent programming
- **Secure**: Sandboxed execution environment
- **High Performance**: Just-In-Time (JIT) compilation optimizes bytecode
- **Robust**: Exception handling and type safety
- **Dynamic**: Reflection and runtime type information

### Java Platform Components

- **Java Development Kit (JDK)**: Complete development environment
- **Java Runtime Environment (JRE)**: Runtime environment for executing Java applications
- **Java Virtual Machine (JVM)**: Executes Java bytecode
- **Java API**: Standard library of classes and interfaces

### Java Editions

- **Java SE (Standard Edition)**: Core Java platform for desktop and server applications
- **Java EE (Enterprise Edition)**: Enterprise features for large-scale applications
- **Java ME (Micro Edition)**: Embedded and mobile applications

### JVM Architecture

```
┌─────────────────────────────────────┐
│           Java Application          │
├─────────────────────────────────────┤
│           Java API                  │
├─────────────────────────────────────┤
│           Java Virtual Machine      │
│  ┌─────────────┬─────────────────┐  │
│  │   Class     │   Runtime Data  │  │
│  │   Loader    │     Areas       │  │
│  ├─────────────┼─────────────────┤  │
│  │   Execution │   Method Area   │  │
│  │   Engine    │   Heap          │  │
│  │             │   Stack         │  │
│  │             │   PC Register   │  │
│  └─────────────┴─────────────────┘  │
├─────────────────────────────────────┤
│           Operating System          │
└─────────────────────────────────────┘
```

---

## Getting Started

### Installation

#### Windows

1. Download JDK from [Oracle](https://www.oracle.com/java/technologies/downloads/) or [AdoptOpenJDK](https://adoptopenjdk.net/)
2. Run the installer and follow the setup wizard
3. Set environment variables:
   - `JAVA_HOME`: Path to JDK installation (e.g., `C:\Program Files\Java\jdk-17`)
   - Add `%JAVA_HOME%\bin` to PATH
4. Verify installation:

```cmd
java -version
javac -version
```

#### macOS

```bash
# Using Homebrew
brew install openjdk@17

# Or download from Oracle/AdoptOpenJDK
# Set JAVA_HOME in ~/.zshrc or ~/.bash_profile
export JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk-17.jdk/Contents/Home
export PATH=$JAVA_HOME/bin:$PATH
```

#### Linux

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install openjdk-17-jdk

# CentOS/RHEL
sudo yum install java-17-openjdk-devel

# Set JAVA_HOME
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk
export PATH=$JAVA_HOME/bin:$PATH
```

### IDE Setup

Popular Java IDEs:

- **IntelliJ IDEA**: Professional IDE with excellent Java support
- **Eclipse**: Free, feature-rich IDE
- **NetBeans**: Free IDE with good Java support
- **VS Code**: Lightweight editor with Java extensions

### First Program

```java
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
```

#### Compilation and Execution

```bash
# Compile
javac HelloWorld.java

# Run
java HelloWorld

# With command line arguments
java HelloWorld arg1 arg2
```

#### Understanding the Program

- `public class HelloWorld`: Class declaration (file name must match class name)
- `public static void main(String[] args)`: Main method entry point
- `System.out.println()`: Print to console with newline
- `String[] args`: Command line arguments array

### Project Structure

```
MyProject/
├── src/
│   └── com/
│       └── example/
│           ├── HelloWorld.java
│           └── utils/
│               └── Helper.java
├── bin/
│   └── com/
│       └── example/
│           ├── HelloWorld.class
│           └── utils/
│               └── Helper.class
├── lib/
│   └── external-library.jar
├── docs/
├── tests/
└── README.md
```

### Build Tools

#### Maven

```xml
<!-- pom.xml -->
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>my-app</artifactId>
    <version>1.0-SNAPSHOT</version>

    <properties>
        <maven.compiler.source>17</maven.compiler.source>
        <maven.compiler.target>17</maven.compiler.target>
    </properties>

    <dependencies>
        <dependency>
            <groupId>junit</groupId>
            <artifactId>junit</artifactId>
            <version>4.13.2</version>
            <scope>test</scope>
        </dependency>
    </dependencies>
</project>
```

#### Gradle

```groovy
// build.gradle
plugins {
    id 'java'
    id 'application'
}

repositories {
    mavenCentral()
}

dependencies {
    testImplementation 'junit:junit:4.13.2'
}

application {
    mainClass = 'com.example.HelloWorld'
}
```

---

## Basic Syntax

### Comments

```java
// Single-line comment
/* Multi-line
   comment */
/**
 * Documentation comment
 */
```

### Variables and Data Types

```java
int age = 25;
double height = 1.75;
char grade = 'A';
boolean isStudent = true;
String name = "John";
```

### Identifiers and Naming Conventions

- Classes: PascalCase (e.g., `MyClass`)
- Variables/Methods: camelCase (e.g., `myVariable`)
- Constants: UPPER_CASE (e.g., `MAX_SIZE`)

### Indentation and Braces

- Code blocks are defined by `{}`

---

## Data Types

### Primitive Types

#### Integer Types

```java
byte b = 127;           // 8-bit, range: -128 to 127
short s = 32767;        // 16-bit, range: -32,768 to 32,767
int i = 2147483647;     // 32-bit, range: -2^31 to 2^31-1
long l = 9223372036854775807L; // 64-bit, range: -2^63 to 2^63-1

// Underscores for readability (Java 7+)
int million = 1_000_000;
long billion = 1_000_000_000L;
```

#### Floating-Point Types

```java
float f = 3.14f;        // 32-bit, 6-7 decimal digits precision
double d = 3.14159265359; // 64-bit, 15-16 decimal digits precision

// Scientific notation
double scientific = 1.23e-4; // 0.000123
```

#### Character Type

```java
char c = 'A';           // 16-bit Unicode character
char unicode = '\u0041'; // Unicode escape sequence
char newline = '\n';     // Escape sequences
```

#### Boolean Type

```java
boolean flag = true;
boolean isActive = false;
```

### Reference Types

#### Classes

```java
String text = "Hello, World!";
Integer number = 42; // Wrapper class
```

#### Arrays

```java
// One-dimensional arrays
int[] numbers = {1, 2, 3, 4, 5};
String[] names = new String[3];
names[0] = "Alice";

// Multi-dimensional arrays
int[][] matrix = {{1, 2, 3}, {4, 5, 6}};
int[][] grid = new int[3][3];
```

#### Enums

```java
public enum DayOfWeek {
    MONDAY, TUESDAY, WEDNESDAY, THURSDAY,
    FRIDAY, SATURDAY, SUNDAY
}

DayOfWeek today = DayOfWeek.MONDAY;
```

### Type Conversion

#### Implicit Casting (Widening)

```java
byte b = 100;
int i = b;      // byte → int (automatic)
long l = i;     // int → long (automatic)
float f = l;    // long → float (automatic)
double d = f;   // float → double (automatic)
```

#### Explicit Casting (Narrowing)

```java
double d = 3.14159;
int i = (int) d;        // double → int (explicit)
long l = 1234567890L;
int n = (int) l;        // long → int (explicit)

// Be careful with data loss
int large = 2000000000;
byte small = (byte) large; // Data loss!
```

#### String Conversion

```java
// Primitive to String
String numStr = String.valueOf(42);
String doubleStr = String.valueOf(3.14);

// String to primitive
int num = Integer.parseInt("42");
double dbl = Double.parseDouble("3.14");
boolean flag = Boolean.parseBoolean("true");

// Using wrapper classes
Integer numObj = Integer.valueOf("42");
Double dblObj = Double.valueOf("3.14");
```

### Wrapper Classes

```java
// Autoboxing (Java 5+)
Integer num = 42;        // int → Integer
Double dbl = 3.14;      // double → Double
Boolean flag = true;     // boolean → Boolean

// Unboxing
int n = num;             // Integer → int
double d = dbl;          // Double → double
boolean b = flag;        // Boolean → boolean

// Null handling
Integer nullable = null;
// int n = nullable;     // NullPointerException!
int safe = nullable != null ? nullable : 0;
```

### Type Checking and instanceof

```java
Object obj = "Hello";
if (obj instanceof String) {
    String str = (String) obj;
    System.out.println(str.length());
}

// Pattern matching (Java 16+)
if (obj instanceof String str) {
    System.out.println(str.length());
}
```

### Type Inference with var (Java 10+)

```java
var message = "Hello, World!"; // String
var numbers = new int[]{1, 2, 3}; // int[]
var list = new ArrayList<String>(); // ArrayList<String>

// Cannot use var for:
// var x; // No initializer
// var y = null; // Cannot infer type
// var z = () -> {}; // Lambda needs explicit type
```

### Null Safety

```java
String text = null;
// System.out.println(text.length()); // NullPointerException

// Safe navigation
if (text != null) {
    System.out.println(text.length());
}

// Optional (Java 8+)
Optional<String> optional = Optional.ofNullable(text);
optional.ifPresent(System.out::println);
```

---

## Control Flow

### Conditional Statements

```java
if (age < 18) {
    System.out.println("Minor");
} else if (age < 65) {
    System.out.println("Adult");
} else {
    System.out.println("Senior");
}

// Ternary operator
String status = (age >= 18) ? "adult" : "minor";
```

### Switch Statement

```java
switch (grade) {
    case 'A':
        System.out.println("Excellent");
        break;
    case 'B':
        System.out.println("Good");
        break;
    default:
        System.out.println("Needs Improvement");
}
```

### Loops

```java
// For loop
for (int i = 0; i < 5; i++) {
    System.out.println(i);
}

// While loop
int count = 0;
while (count < 5) {
    System.out.println(count);
    count++;
}

// Do-while loop
int n = 0;
do {
    System.out.println(n);
    n++;
} while (n < 5);
```

---

## Methods

### Defining and Calling Methods

```java
public static int add(int a, int b) {
    return a + b;
}

int result = add(5, 3);
```

### Method Overloading

```java
public static int add(int a, int b) {
    return a + b;
}
public static double add(double a, double b) {
    return a + b;
}
```

---

## Object-Oriented Programming

### Classes and Objects

```java
public class Person {
    String name;
    int age;

    public Person(String name, int age) {
        this.name = name;
        this.age = age;
    }

    public void greet() {
        System.out.println("Hello, my name is " + name);
    }
}

Person p = new Person("Alice", 25);
p.greet();
```

### Inheritance

```java
public class Animal {
    public void speak() {
        System.out.println("Animal speaks");
    }
}

public class Dog extends Animal {
    @Override
    public void speak() {
        System.out.println("Dog barks");
    }
}

Dog d = new Dog();
d.speak();
```

### Interfaces

```java
public interface Drawable {
    void draw();
}

public class Circle implements Drawable {
    public void draw() {
        System.out.println("Drawing Circle");
    }
}
```

### Abstract Classes

```java
public abstract class Shape {
    abstract double area();
}

public class Rectangle extends Shape {
    double width, height;
    Rectangle(double w, double h) { width = w; height = h; }
    double area() { return width * height; }
}
```

### Encapsulation

```java
public class Account {
    private double balance;
    public double getBalance() { return balance; }
    public void deposit(double amount) { balance += amount; }
}
```

### Polymorphism

- Method Overriding
- Interface Implementation

---

## Packages and Imports

```java
package com.example;
import java.util.*;
```

- Organize code into namespaces

---

## Exception Handling

```java
try {
    int result = 10 / 0;
} catch (ArithmeticException e) {
    System.out.println("Cannot divide by zero");
} finally {
    System.out.println("Cleanup");
}

// Custom Exception
class MyException extends Exception {
    public MyException(String message) { super(message); }
}
```

---

## File I/O

```java
import java.io.*;

// Reading a file
BufferedReader reader = new BufferedReader(new FileReader("file.txt"));
String line;
while ((line = reader.readLine()) != null) {
    System.out.println(line);
}
reader.close();

// Writing to a file
BufferedWriter writer = new BufferedWriter(new FileWriter("output.txt"));
writer.write("Hello, World!\n");
writer.close();
```

---

## Collections Framework

The Java Collections Framework provides a unified architecture for representing and manipulating collections of objects.

### Core Interfaces

#### Collection Interface

```java
import java.util.*;

Collection<String> collection = new ArrayList<>();
collection.add("item");
collection.remove("item");
collection.contains("item");
collection.size();
collection.clear();
```

#### List Interface

```java
// ArrayList - Dynamic array, fast random access
List<String> arrayList = new ArrayList<>();
arrayList.add("first");
arrayList.add(0, "inserted"); // Insert at index
arrayList.set(1, "updated");  // Update at index
String item = arrayList.get(0); // Get by index
arrayList.remove(0);           // Remove by index
arrayList.remove("item");      // Remove by value

// LinkedList - Doubly linked list, fast insertions/deletions
List<String> linkedList = new LinkedList<>();
linkedList.addFirst("first");
linkedList.addLast("last");
String first = linkedList.getFirst();
String last = linkedList.getLast();

// Vector - Thread-safe dynamic array (legacy)
Vector<String> vector = new Vector<>();
vector.addElement("item");
```

#### Set Interface

```java
// HashSet - Unordered, no duplicates, O(1) operations
Set<String> hashSet = new HashSet<>();
hashSet.add("apple");
hashSet.add("banana");
hashSet.add("apple"); // Duplicate ignored
System.out.println(hashSet.size()); // 2

// LinkedHashSet - Ordered HashSet
Set<String> linkedHashSet = new LinkedHashSet<>();
linkedHashSet.add("first");
linkedHashSet.add("second");
// Maintains insertion order

// TreeSet - Sorted set
Set<String> treeSet = new TreeSet<>();
treeSet.add("zebra");
treeSet.add("apple");
treeSet.add("banana");
// Automatically sorted: [apple, banana, zebra]
```

#### Map Interface

```java
// HashMap - Key-value pairs, no order
Map<String, Integer> hashMap = new HashMap<>();
hashMap.put("apple", 1);
hashMap.put("banana", 2);
hashMap.put("apple", 3); // Overwrites previous value
Integer value = hashMap.get("apple"); // 3
hashMap.containsKey("apple"); // true
hashMap.containsValue(3); // true

// LinkedHashMap - Ordered HashMap
Map<String, Integer> linkedHashMap = new LinkedHashMap<>();
// Maintains insertion order

// TreeMap - Sorted map
Map<String, Integer> treeMap = new TreeMap<>();
treeMap.put("zebra", 3);
treeMap.put("apple", 1);
treeMap.put("banana", 2);
// Automatically sorted by keys

// Hashtable - Thread-safe map (legacy)
Hashtable<String, Integer> hashtable = new Hashtable<>();
```

### Queue Interface

```java
// PriorityQueue - Priority-based queue
Queue<Integer> priorityQueue = new PriorityQueue<>();
priorityQueue.offer(5);
priorityQueue.offer(1);
priorityQueue.offer(3);
System.out.println(priorityQueue.poll()); // 1 (smallest first)

// LinkedList as Queue
Queue<String> queue = new LinkedList<>();
queue.offer("first");
queue.offer("second");
String head = queue.peek(); // View head without removing
String removed = queue.poll(); // Remove and return head
```

### Deque Interface

```java
// ArrayDeque - Double-ended queue
Deque<String> deque = new ArrayDeque<>();
deque.addFirst("first");
deque.addLast("last");
deque.offerFirst("newFirst");
deque.offerLast("newLast");
String first = deque.getFirst();
String last = deque.getLast();
```

### Collection Operations

```java
List<String> list = Arrays.asList("apple", "banana", "cherry");

// Iteration
for (String item : list) {
    System.out.println(item);
}

// Using Iterator
Iterator<String> iterator = list.iterator();
while (iterator.hasNext()) {
    String item = iterator.next();
    System.out.println(item);
}

// Using forEach (Java 8+)
list.forEach(System.out::println);

// Using streams (Java 8+)
list.stream()
    .filter(item -> item.startsWith("a"))
    .map(String::toUpperCase)
    .forEach(System.out::println);
```

### Collections Utility Class

```java
List<String> list = new ArrayList<>();
Collections.addAll(list, "apple", "banana", "cherry");
Collections.sort(list); // Natural ordering
Collections.reverse(list);
Collections.shuffle(list);
Collections.rotate(list, 1);
Collections.fill(list, "default");

// Synchronized collections
List<String> syncList = Collections.synchronizedList(new ArrayList<>());
Map<String, Integer> syncMap = Collections.synchronizedMap(new HashMap<>());

// Unmodifiable collections
List<String> unmodifiableList = Collections.unmodifiableList(list);
Map<String, Integer> unmodifiableMap = Collections.unmodifiableMap(new HashMap<>());
```

### Custom Collections

```java
// Custom comparator
Comparator<String> lengthComparator = (s1, s2) ->
    Integer.compare(s1.length(), s2.length());

List<String> words = Arrays.asList("cat", "dog", "elephant");
Collections.sort(words, lengthComparator);

// Custom comparable
class Person implements Comparable<Person> {
    private String name;
    private int age;

    public Person(String name, int age) {
        this.name = name;
        this.age = age;
    }

    @Override
    public int compareTo(Person other) {
        return Integer.compare(this.age, other.age);
    }

    // getters, toString, etc.
}
```

### Collection Performance Characteristics

| Collection | Get      | Add      | Remove   | Contains | Order |
| ---------- | -------- | -------- | -------- | -------- | ----- |
| ArrayList  | O(1)     | O(1)     | O(n)     | O(n)     | Yes   |
| LinkedList | O(n)     | O(1)     | O(1)     | O(n)     | Yes   |
| HashSet    | O(1)     | O(1)     | O(1)     | O(1)     | No    |
| TreeSet    | O(log n) | O(log n) | O(log n) | O(log n) | Yes   |
| HashMap    | O(1)     | O(1)     | O(1)     | O(1)     | No    |
| TreeMap    | O(log n) | O(log n) | O(log n) | O(log n) | Yes   |

---

## Generics

```java
public class Box<T> {
    private T value;
    public void set(T value) { this.value = value; }
    public T get() { return value; }
}

Box<Integer> intBox = new Box<>();
intBox.set(123);
```

---

## Multithreading

```java
class MyThread extends Thread {
    public void run() {
        System.out.println("Thread running");
    }
}
MyThread t = new MyThread();
t.start();

// Runnable
Runnable r = () -> System.out.println("Runnable running");
new Thread(r).start();
```

---

## Lambda Expressions and Streams

### Lambda Expressions (Java 8+)

Lambda expressions provide a concise way to implement functional interfaces.

#### Functional Interfaces

```java
// Single abstract method interface
@FunctionalInterface
interface MathOperation {
    int operate(int a, int b);
}

// Using lambda
MathOperation add = (a, b) -> a + b;
MathOperation multiply = (a, b) -> a * b;

System.out.println(add.operate(5, 3));      // 8
System.out.println(multiply.operate(4, 6)); // 24
```

#### Built-in Functional Interfaces

```java
import java.util.function.*;

// Predicate - boolean test
Predicate<String> isEmpty = String::isEmpty;
Predicate<String> hasLength = s -> s.length() > 5;

// Function - transform input to output
Function<String, Integer> getLength = String::length;
Function<Integer, String> toString = Object::toString;

// Consumer - consume input, no output
Consumer<String> printer = System.out::println;
Consumer<String> upperPrinter = s -> System.out.println(s.toUpperCase());

// Supplier - provide output, no input
Supplier<String> greeting = () -> "Hello, World!";
Supplier<Double> random = Math::random;

// BiFunction - two inputs, one output
BiFunction<String, String, String> concat = String::concat;
BiFunction<Integer, Integer, Integer> sum = Integer::sum;
```

#### Method References

```java
List<String> names = Arrays.asList("Alice", "Bob", "Charlie");

// Static method reference
names.forEach(System.out::println);

// Instance method reference
names.stream()
     .map(String::toUpperCase)
     .forEach(System.out::println);

// Constructor reference
Supplier<ArrayList<String>> listSupplier = ArrayList::new;
```

### Streams API

#### Creating Streams

```java
// From collections
List<String> list = Arrays.asList("apple", "banana", "cherry");
Stream<String> stream1 = list.stream();

// From arrays
String[] array = {"apple", "banana", "cherry"};
Stream<String> stream2 = Arrays.stream(array);

// From individual elements
Stream<String> stream3 = Stream.of("apple", "banana", "cherry");

// Infinite streams
Stream<Integer> infinite = Stream.iterate(0, n -> n + 1);
Stream<Double> random = Stream.generate(Math::random);
```

#### Stream Operations

```java
List<String> fruits = Arrays.asList("apple", "banana", "cherry", "date", "elderberry");

// Filtering
fruits.stream()
      .filter(fruit -> fruit.length() > 5)
      .forEach(System.out::println);

// Mapping
fruits.stream()
      .map(String::toUpperCase)
      .forEach(System.out::println);

// Sorting
fruits.stream()
      .sorted()
      .forEach(System.out::println);

// Limiting
fruits.stream()
      .limit(3)
      .forEach(System.out::println);

// Skipping
fruits.stream()
      .skip(2)
      .forEach(System.out::println);
```

#### Terminal Operations

```java
List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

// Collecting
List<Integer> evenNumbers = numbers.stream()
    .filter(n -> n % 2 == 0)
    .collect(Collectors.toList());

// Reducing
int sum = numbers.stream()
    .reduce(0, Integer::sum);

// Finding
Optional<Integer> firstEven = numbers.stream()
    .filter(n -> n % 2 == 0)
    .findFirst();

// Matching
boolean allPositive = numbers.stream()
    .allMatch(n -> n > 0);

boolean anyEven = numbers.stream()
    .anyMatch(n -> n % 2 == 0);

boolean noneNegative = numbers.stream()
    .noneMatch(n -> n < 0);
```

#### Parallel Streams

```java
List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

// Parallel processing
int sum = numbers.parallelStream()
    .filter(n -> n % 2 == 0)
    .mapToInt(Integer::intValue)
    .sum();

// Custom thread pool
ForkJoinPool customThreadPool = new ForkJoinPool(4);
int result = customThreadPool.submit(() ->
    numbers.parallelStream()
           .filter(n -> n % 2 == 0)
           .mapToInt(Integer::intValue)
           .sum()
).get();
```

#### Advanced Stream Operations

```java
List<Person> people = Arrays.asList(
    new Person("Alice", 25),
    new Person("Bob", 30),
    new Person("Charlie", 35)
);

// Grouping
Map<Integer, List<Person>> byAge = people.stream()
    .collect(Collectors.groupingBy(Person::getAge));

// Partitioning
Map<Boolean, List<Person>> byAgeGroup = people.stream()
    .collect(Collectors.partitioningBy(p -> p.getAge() < 30));

// Joining
String names = people.stream()
    .map(Person::getName)
    .collect(Collectors.joining(", "));

// Statistics
IntSummaryStatistics stats = people.stream()
    .mapToInt(Person::getAge)
    .summaryStatistics();
```

---

## Annotations

Annotations provide metadata about program elements and can be processed at compile time or runtime.

### Built-in Annotations

```java
// @Override - Indicates method overrides superclass method
@Override
public String toString() {
    return "Custom toString";
}

// @Deprecated - Marks element as deprecated
@Deprecated
public void oldMethod() {
    // Old implementation
}

// @SuppressWarnings - Suppresses compiler warnings
@SuppressWarnings("unchecked")
public void uncheckedMethod() {
    List list = new ArrayList();
}

// @FunctionalInterface - Marks interface as functional
@FunctionalInterface
interface MyFunctionalInterface {
    void doSomething();
}

// @SafeVarargs - Indicates varargs method is safe
@SafeVarargs
public static <T> List<T> asList(T... elements) {
    return Arrays.asList(elements);
}
```

### Creating Custom Annotations

```java
// Simple annotation
@interface MyAnnotation {
    String value() default "default";
    int count() default 0;
}

// Annotation with retention policy
@Retention(RetentionPolicy.RUNTIME)
@Target({ElementType.METHOD, ElementType.TYPE})
@interface TestAnnotation {
    String description();
    int priority() default 1;
}

// Using custom annotation
@TestAnnotation(description = "Test method", priority = 2)
public void testMethod() {
    // Test implementation
}
```

### Annotation Processing

```java
// Runtime annotation processing
Method[] methods = MyClass.class.getMethods();
for (Method method : methods) {
    if (method.isAnnotationPresent(TestAnnotation.class)) {
        TestAnnotation annotation = method.getAnnotation(TestAnnotation.class);
        System.out.println("Method: " + method.getName());
        System.out.println("Description: " + annotation.description());
        System.out.println("Priority: " + annotation.priority());
    }
}
```

---

## Reflection

Reflection allows examining and modifying the behavior of classes, interfaces, fields, and methods at runtime.

### Class Information

```java
// Getting class information
Class<?> clazz = String.class;
System.out.println("Class name: " + clazz.getName());
System.out.println("Simple name: " + clazz.getSimpleName());
System.out.println("Package: " + clazz.getPackage());
System.out.println("Superclass: " + clazz.getSuperclass());
System.out.println("Interfaces: " + Arrays.toString(clazz.getInterfaces()));

// Checking modifiers
int modifiers = clazz.getModifiers();
System.out.println("Is public: " + Modifier.isPublic(modifiers));
System.out.println("Is final: " + Modifier.isFinal(modifiers));
```

### Field Access

```java
class Person {
    private String name;
    public int age;

    public Person(String name, int age) {
        this.name = name;
        this.age = age;
    }
}

// Accessing fields
Person person = new Person("Alice", 25);
Class<?> clazz = person.getClass();

// Get all fields
Field[] fields = clazz.getDeclaredFields();
for (Field field : fields) {
    System.out.println("Field: " + field.getName());
    System.out.println("Type: " + field.getType());
    System.out.println("Modifiers: " + Modifier.toString(field.getModifiers()));
}

// Access private field
try {
    Field nameField = clazz.getDeclaredField("name");
    nameField.setAccessible(true);
    String name = (String) nameField.get(person);
    System.out.println("Name: " + name);

    // Modify private field
    nameField.set(person, "Bob");
    System.out.println("New name: " + nameField.get(person));
} catch (Exception e) {
    e.printStackTrace();
}
```

### Method Invocation

```java
class Calculator {
    public int add(int a, int b) {
        return a + b;
    }

    private int multiply(int a, int b) {
        return a * b;
    }
}

Calculator calc = new Calculator();
Class<?> clazz = calc.getClass();

// Invoke public method
try {
    Method addMethod = clazz.getMethod("add", int.class, int.class);
    int result = (int) addMethod.invoke(calc, 5, 3);
    System.out.println("Result: " + result);
} catch (Exception e) {
    e.printStackTrace();
}

// Invoke private method
try {
    Method multiplyMethod = clazz.getDeclaredMethod("multiply", int.class, int.class);
    multiplyMethod.setAccessible(true);
    int result = (int) multiplyMethod.invoke(calc, 4, 6);
    System.out.println("Multiply result: " + result);
} catch (Exception e) {
    e.printStackTrace();
}
```

### Constructor Access

```java
// Get constructors
Constructor<?>[] constructors = clazz.getConstructors();
for (Constructor<?> constructor : constructors) {
    System.out.println("Constructor: " + constructor);
    System.out.println("Parameters: " + Arrays.toString(constructor.getParameterTypes()));
}

// Create instance using reflection
try {
    Constructor<Person> constructor = Person.class.getConstructor(String.class, int.class);
    Person newPerson = constructor.newInstance("Charlie", 30);
    System.out.println("Created: " + newPerson);
} catch (Exception e) {
    e.printStackTrace();
}
```

---

## Serialization

Serialization is the process of converting objects to a byte stream for storage or transmission.

### Basic Serialization

```java
import java.io.*;

class Person implements Serializable {
    private static final long serialVersionUID = 1L;
    private String name;
    private int age;

    public Person(String name, int age) {
        this.name = name;
        this.age = age;
    }

    // getters, setters, toString
}

// Serialize object
Person person = new Person("Alice", 25);
try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("person.ser"))) {
    oos.writeObject(person);
    System.out.println("Object serialized successfully");
} catch (IOException e) {
    e.printStackTrace();
}

// Deserialize object
try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream("person.ser"))) {
    Person deserializedPerson = (Person) ois.readObject();
    System.out.println("Deserialized: " + deserializedPerson);
} catch (IOException | ClassNotFoundException e) {
    e.printStackTrace();
}
```

### Custom Serialization

```java
class CustomPerson implements Serializable {
    private String name;
    private transient int age; // Won't be serialized

    public CustomPerson(String name, int age) {
        this.name = name;
        this.age = age;
    }

    // Custom serialization
    private void writeObject(ObjectOutputStream out) throws IOException {
        out.defaultWriteObject();
        out.writeInt(age * 2); // Store age * 2
    }

    // Custom deserialization
    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        this.age = in.readInt() / 2; // Restore original age
    }
}
```

### JSON Serialization

```java
// Using Jackson library
import com.fasterxml.jackson.databind.ObjectMapper;

ObjectMapper mapper = new ObjectMapper();

// Serialize to JSON
Person person = new Person("Alice", 25);
String json = mapper.writeValueAsString(person);
System.out.println("JSON: " + json);

// Deserialize from JSON
Person deserializedPerson = mapper.readValue(json, Person.class);
System.out.println("Deserialized: " + deserializedPerson);
```

---

## Networking

Java provides comprehensive networking capabilities through the `java.net` package.

### URL Handling

```java
import java.net.*;

// URL parsing
URL url = new URL("https://api.example.com/users?id=123");
System.out.println("Protocol: " + url.getProtocol());
System.out.println("Host: " + url.getHost());
System.out.println("Port: " + url.getPort());
System.out.println("Path: " + url.getPath());
System.out.println("Query: " + url.getQuery());

// URL connection
URLConnection connection = url.openConnection();
connection.setRequestProperty("User-Agent", "Java Client");
InputStream input = connection.getInputStream();
// Read response...
```

### HTTP Client (Java 11+)

```java
import java.net.http.*;
import java.net.URI;

// GET request
HttpClient client = HttpClient.newHttpClient();
HttpRequest request = HttpRequest.newBuilder()
    .uri(URI.create("https://api.example.com/users"))
    .build();

HttpResponse<String> response = client.send(request,
    HttpResponse.BodyHandlers.ofString());
System.out.println("Response: " + response.body());

// POST request
String jsonBody = "{\"name\":\"Alice\",\"age\":25}";
HttpRequest postRequest = HttpRequest.newBuilder()
    .uri(URI.create("https://api.example.com/users"))
    .header("Content-Type", "application/json")
    .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
    .build();

HttpResponse<String> postResponse = client.send(postRequest,
    HttpResponse.BodyHandlers.ofString());
```

### Socket Programming

```java
// Server
public class SimpleServer {
    public static void main(String[] args) {
        try (ServerSocket serverSocket = new ServerSocket(8080)) {
            System.out.println("Server listening on port 8080");

            while (true) {
                Socket clientSocket = serverSocket.accept();
                System.out.println("Client connected: " + clientSocket.getInetAddress());

                // Handle client in separate thread
                new Thread(() -> handleClient(clientSocket)).start();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void handleClient(Socket clientSocket) {
        try (BufferedReader in = new BufferedReader(
                new InputStreamReader(clientSocket.getInputStream()));
             PrintWriter out = new PrintWriter(clientSocket.getOutputStream(), true)) {

            String inputLine;
            while ((inputLine = in.readLine()) != null) {
                System.out.println("Received: " + inputLine);
                out.println("Echo: " + inputLine);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

// Client
public class SimpleClient {
    public static void main(String[] args) {
        try (Socket socket = new Socket("localhost", 8080);
             PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
             BufferedReader in = new BufferedReader(
                 new InputStreamReader(socket.getInputStream()));
             BufferedReader stdIn = new BufferedReader(
                 new InputStreamReader(System.in))) {

            String userInput;
            while ((userInput = stdIn.readLine()) != null) {
                out.println(userInput);
                String response = in.readLine();
                System.out.println("Server response: " + response);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
```

---

## Database Connectivity

Java Database Connectivity (JDBC) provides a standard API for database operations.

### Basic JDBC Operations

```java
import java.sql.*;

// Database connection
String url = "jdbc:mysql://localhost:3306/mydb";
String username = "user";
String password = "password";

try (Connection connection = DriverManager.getConnection(url, username, password)) {
    System.out.println("Database connected successfully");

    // Create table
    String createTable = """
        CREATE TABLE IF NOT EXISTS users (
            id INT PRIMARY KEY AUTO_INCREMENT,
            name VARCHAR(100) NOT NULL,
            email VARCHAR(100) UNIQUE,
            age INT
        )
    """;

    try (Statement stmt = connection.createStatement()) {
        stmt.execute(createTable);
    }

    // Insert data
    String insert = "INSERT INTO users (name, email, age) VALUES (?, ?, ?)";
    try (PreparedStatement pstmt = connection.prepareStatement(insert)) {
        pstmt.setString(1, "Alice");
        pstmt.setString(2, "alice@example.com");
        pstmt.setInt(3, 25);
        pstmt.executeUpdate();
    }

    // Query data
    String select = "SELECT * FROM users WHERE age > ?";
    try (PreparedStatement pstmt = connection.prepareStatement(select)) {
        pstmt.setInt(1, 20);
        ResultSet rs = pstmt.executeQuery();

        while (rs.next()) {
            int id = rs.getInt("id");
            String name = rs.getString("name");
            String email = rs.getString("email");
            int age = rs.getInt("age");
            System.out.printf("ID: %d, Name: %s, Email: %s, Age: %d%n",
                id, name, email, age);
        }
    }

} catch (SQLException e) {
    e.printStackTrace();
}
```

### Connection Pooling

```java
// Using HikariCP
import com.zaxxer.hikari.HikariConfig;
import com.zaxxer.hikari.HikariDataSource;

HikariConfig config = new HikariConfig();
config.setJdbcUrl("jdbc:mysql://localhost:3306/mydb");
config.setUsername("user");
config.setPassword("password");
config.setMaximumPoolSize(10);
config.setMinimumIdle(5);

HikariDataSource dataSource = new HikariDataSource(config);

try (Connection connection = dataSource.getConnection()) {
    // Use connection
} catch (SQLException e) {
    e.printStackTrace();
}
```

### Transaction Management

```java
try (Connection connection = dataSource.getConnection()) {
    connection.setAutoCommit(false);

    try {
        // First operation
        String update1 = "UPDATE accounts SET balance = balance - ? WHERE id = ?";
        try (PreparedStatement pstmt = connection.prepareStatement(update1)) {
            pstmt.setBigDecimal(1, new BigDecimal("100"));
            pstmt.setInt(2, 1);
            pstmt.executeUpdate();
        }

        // Second operation
        String update2 = "UPDATE accounts SET balance = balance + ? WHERE id = ?";
        try (PreparedStatement pstmt = connection.prepareStatement(update2)) {
            pstmt.setBigDecimal(1, new BigDecimal("100"));
            pstmt.setInt(2, 2);
            pstmt.executeUpdate();
        }

        connection.commit();
        System.out.println("Transaction committed successfully");

    } catch (SQLException e) {
        connection.rollback();
        System.out.println("Transaction rolled back");
        throw e;
    }
} catch (SQLException e) {
    e.printStackTrace();
}
```

---

## Best Practices

### Code Style and Naming Conventions

```java
// Class names: PascalCase
public class UserAccount {
    // Constants: UPPER_CASE with underscores
    public static final int MAX_LOGIN_ATTEMPTS = 3;

    // Instance variables: camelCase
    private String userName;
    private int loginAttempts;

    // Method names: camelCase
    public void validateUser() {
        // Method implementation
    }

    // Boolean methods: is/has/can prefix
    public boolean isValidUser() {
        return loginAttempts < MAX_LOGIN_ATTEMPTS;
    }
}
```

### Method Design

```java
// Keep methods short and focused (single responsibility)
public class UserService {
    // Good: Single responsibility
    public User createUser(String name, String email) {
        validateUserData(name, email);
        User user = new User(name, email);
        saveUser(user);
        return user;
    }

    private void validateUserData(String name, String email) {
        if (name == null || name.trim().isEmpty()) {
            throw new IllegalArgumentException("Name cannot be empty");
        }
        if (email == null || !email.contains("@")) {
            throw new IllegalArgumentException("Invalid email format");
        }
    }

    private void saveUser(User user) {
        // Database operation
    }
}
```

### Exception Handling

```java
// Use specific exceptions
public class FileProcessor {
    public void processFile(String filename) {
        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
            String line;
            while ((line = reader.readLine()) != null) {
                processLine(line);
            }
        } catch (FileNotFoundException e) {
            logger.error("File not found: " + filename, e);
            throw new ProcessingException("Cannot process file: " + filename, e);
        } catch (IOException e) {
            logger.error("Error reading file: " + filename, e);
            throw new ProcessingException("Error processing file: " + filename, e);
        }
    }

    private void processLine(String line) {
        // Process individual line
    }
}
```

### Resource Management

```java
// Use try-with-resources for automatic cleanup
public class DatabaseConnection {
    public void executeQuery(String sql) {
        try (Connection conn = dataSource.getConnection();
             PreparedStatement stmt = conn.prepareStatement(sql);
             ResultSet rs = stmt.executeQuery()) {

            while (rs.next()) {
                processResult(rs);
            }
        } catch (SQLException e) {
            throw new DatabaseException("Query execution failed", e);
        }
    }
}
```

### Immutability

```java
// Make classes immutable when possible
public final class ImmutablePoint {
    private final double x;
    private final double y;

    public ImmutablePoint(double x, double y) {
        this.x = x;
        this.y = y;
    }

    public double getX() { return x; }
    public double getY() { return y; }

    // Return new instance instead of modifying
    public ImmutablePoint translate(double dx, double dy) {
        return new ImmutablePoint(x + dx, y + dy);
    }
}
```

### Documentation

```java
/**
 * Represents a user account in the system.
 *
 * @author John Doe
 * @version 1.0
 * @since 1.0
 */
public class UserAccount {
    /**
     * The maximum number of login attempts allowed.
     */
    private static final int MAX_LOGIN_ATTEMPTS = 3;

    /**
     * Creates a new user account with the specified credentials.
     *
     * @param username the username for the account
     * @param email the email address for the account
     * @throws IllegalArgumentException if username or email is null or empty
     * @throws DuplicateUserException if a user with the same username already exists
     */
    public UserAccount(String username, String email) {
        // Constructor implementation
    }

    /**
     * Validates the user's login credentials.
     *
     * @param password the password to validate
     * @return true if the password is correct, false otherwise
     */
    public boolean validatePassword(String password) {
        // Implementation
        return false;
    }
}
```

### Performance Considerations

```java
// Use StringBuilder for string concatenation in loops
public class StringProcessor {
    public String buildLargeString(List<String> items) {
        StringBuilder sb = new StringBuilder();
        for (String item : items) {
            sb.append(item).append("\n");
        }
        return sb.toString();
    }

    // Use appropriate collection types
    public void processData() {
        // Use ArrayList for random access
        List<String> randomAccessList = new ArrayList<>();

        // Use LinkedList for frequent insertions/deletions
        List<String> frequentModifications = new LinkedList<>();

        // Use HashSet for unique elements, no order needed
        Set<String> uniqueItems = new HashSet<>();

        // Use TreeSet for sorted unique elements
        Set<String> sortedUniqueItems = new TreeSet<>();
    }
}
```

### Thread Safety

```java
// Use thread-safe collections when needed
public class ThreadSafeCounter {
    private final AtomicInteger counter = new AtomicInteger(0);

    public int increment() {
        return counter.incrementAndGet();
    }

    public int getValue() {
        return counter.get();
    }
}

// Use synchronized blocks for critical sections
public class BankAccount {
    private double balance;
    private final Object lock = new Object();

    public void deposit(double amount) {
        synchronized (lock) {
            if (amount > 0) {
                balance += amount;
            }
        }
    }

    public double getBalance() {
        synchronized (lock) {
            return balance;
        }
    }
}
```

---

## Common Libraries

### Core Java Libraries

#### java.util Package

```java
import java.util.*;

// Random number generation
Random random = new Random();
int randomNumber = random.nextInt(100); // 0-99
double randomDouble = random.nextDouble(); // 0.0-1.0

// Date and Calendar (legacy)
Date now = new Date();
Calendar cal = Calendar.getInstance();
cal.set(2023, Calendar.DECEMBER, 25);

// Scanner for input
Scanner scanner = new Scanner(System.in);
String input = scanner.nextLine();
int number = scanner.nextInt();
```

#### java.time Package (Java 8+)

```java
import java.time.*;
import java.time.format.*;

// Current date/time
LocalDate today = LocalDate.now();
LocalTime now = LocalTime.now();
LocalDateTime dateTime = LocalDateTime.now();

// Creating specific dates
LocalDate birthday = LocalDate.of(1990, Month.JANUARY, 15);
LocalTime meeting = LocalTime.of(14, 30); // 2:30 PM

// Date arithmetic
LocalDate tomorrow = today.plusDays(1);
LocalDate nextWeek = today.plusWeeks(1);
LocalDate nextMonth = today.plusMonths(1);

// Duration and Period
Duration duration = Duration.between(meeting, LocalTime.now());
Period period = Period.between(birthday, today);

// Formatting
DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
String formatted = dateTime.format(formatter);

// Parsing
LocalDateTime parsed = LocalDateTime.parse("2023-12-25 14:30:00", formatter);
```

#### java.io Package

```java
import java.io.*;

// File operations
File file = new File("data.txt");
if (file.exists()) {
    System.out.println("File size: " + file.length());
    System.out.println("Last modified: " + new Date(file.lastModified()));
}

// Directory operations
File dir = new File("mydir");
if (!dir.exists()) {
    dir.mkdir();
}

// List directory contents
File[] files = dir.listFiles();
for (File f : files) {
    System.out.println(f.getName() + " - " + f.length() + " bytes");
}
```

#### java.nio Package

```java
import java.nio.file.*;
import java.nio.charset.*;

// File operations with NIO
Path path = Paths.get("data.txt");
Files.write(path, "Hello, World!".getBytes());

// Read all lines
List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8);

// Copy file
Files.copy(Paths.get("source.txt"), Paths.get("dest.txt"));

// Walk directory tree
Files.walk(Paths.get("."))
     .filter(Files::isRegularFile)
     .forEach(System.out::println);
```

### Third-Party Libraries

#### Apache Commons

```java
// Apache Commons Lang
import org.apache.commons.lang3.*;

StringUtils.isBlank(""); // true
StringUtils.reverse("hello"); // "olleh"
StringUtils.join(Arrays.asList("a", "b", "c"), ","); // "a,b,c"

// Apache Commons IO
import org.apache.commons.io.*;

String content = FileUtils.readFileToString(new File("data.txt"), "UTF-8");
FileUtils.copyFile(new File("source.txt"), new File("dest.txt"));
```

#### Google Guava

```java
import com.google.common.collect.*;
import com.google.common.base.*;

// Collections
List<String> list = Lists.newArrayList("a", "b", "c");
Set<String> set = Sets.newHashSet("a", "b", "c");
Map<String, Integer> map = Maps.newHashMap();

// Optional (before Java 8)
Optional<String> optional = Optional.of("value");
if (optional.isPresent()) {
    System.out.println(optional.get());
}

// Preconditions
Preconditions.checkNotNull(user, "User cannot be null");
Preconditions.checkArgument(age > 0, "Age must be positive");
```

#### Jackson (JSON Processing)

```java
import com.fasterxml.jackson.databind.*;

ObjectMapper mapper = new ObjectMapper();

// Serialize to JSON
User user = new User("Alice", 25);
String json = mapper.writeValueAsString(user);

// Deserialize from JSON
User deserializedUser = mapper.readValue(json, User.class);

// Pretty printing
String prettyJson = mapper.writerWithDefaultPrettyPrinter()
                         .writeValueAsString(user);
```

#### Logging Frameworks

```java
// SLF4J with Logback
import org.slf4j.*;

Logger logger = LoggerFactory.getLogger(MyClass.class);
logger.info("Application started");
logger.error("An error occurred", exception);

// Log4j2
import org.apache.logging.log4j.*;

Logger logger = LogManager.getLogger(MyClass.class);
logger.info("Application started");
logger.error("An error occurred", exception);
```

#### Testing Libraries

```java
// JUnit 5
import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

@Test
void testAddition() {
    Calculator calc = new Calculator();
    assertEquals(4, calc.add(2, 2));
}

@ParameterizedTest
@ValueSource(ints = {1, 2, 3, 4, 5})
void testIsPositive(int number) {
    assertTrue(number > 0);
}

// Mockito
import org.mockito.*;
import static org.mockito.Mockito.*;

UserService userService = mock(UserService.class);
when(userService.getUser(1L)).thenReturn(new User("Alice", 25));

User user = userService.getUser(1L);
verify(userService).getUser(1L);
```

#### Build Tools

##### Maven

```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>my-app</artifactId>
    <version>1.0-SNAPSHOT</version>

    <properties>
        <maven.compiler.source>17</maven.compiler.source>
        <maven.compiler.target>17</maven.compiler.target>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
    </properties>

    <dependencies>
        <dependency>
            <groupId>junit</groupId>
            <artifactId>junit</artifactId>
            <version>4.13.2</version>
            <scope>test</scope>
        </dependency>
        <dependency>
            <groupId>org.slf4j</groupId>
            <artifactId>slf4j-api</artifactId>
            <version>1.7.36</version>
        </dependency>
    </dependencies>

    <build>
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-compiler-plugin</artifactId>
                <version>3.11.0</version>
            </plugin>
        </plugins>
    </build>
</project>
```

##### Gradle

```groovy
plugins {
    id 'java'
    id 'application'
    id 'jacoco'
}

repositories {
    mavenCentral()
}

dependencies {
    implementation 'org.slf4j:slf4j-api:1.7.36'
    testImplementation 'org.junit.jupiter:junit-jupiter:5.9.2'
    testImplementation 'org.mockito:mockito-core:5.3.1'
}

application {
    mainClass = 'com.example.Main'
}

test {
    useJUnitPlatform()
}

jacocoTestReport {
    reports {
        xml.required = true
        html.required = true
    }
}
```

---

## Debugging and Testing

### Debugging Techniques

#### IDE Debugging

```java
// Set breakpoints in your IDE
public class DebugExample {
    public static void main(String[] args) {
        int sum = 0;
        for (int i = 1; i <= 10; i++) {
            sum += i; // Set breakpoint here
            System.out.println("Sum: " + sum);
        }
    }
}
```

#### Logging for Debugging

```java
import java.util.logging.*;

public class DebugLogger {
    private static final Logger logger = Logger.getLogger(DebugLogger.class.getName());

    public void processData(String data) {
        logger.info("Processing data: " + data);

        try {
            // Process data
            logger.fine("Data processed successfully");
        } catch (Exception e) {
            logger.severe("Error processing data: " + e.getMessage());
            throw e;
        }
    }
}
```

#### Assertions

```java
public class AssertionExample {
    public int divide(int a, int b) {
        assert b != 0 : "Division by zero";
        return a / b;
    }

    public void processArray(int[] array) {
        assert array != null : "Array cannot be null";
        assert array.length > 0 : "Array cannot be empty";

        // Process array
    }
}
```

### Unit Testing

#### JUnit 5 Testing

```java
import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

class CalculatorTest {
    private Calculator calculator;

    @BeforeEach
    void setUp() {
        calculator = new Calculator();
    }

    @Test
    void testAddition() {
        assertEquals(4, calculator.add(2, 2));
        assertEquals(0, calculator.add(-2, 2));
        assertEquals(-4, calculator.add(-2, -2));
    }

    @Test
    void testDivision() {
        assertEquals(2.0, calculator.divide(4, 2));
        assertThrows(ArithmeticException.class, () -> calculator.divide(4, 0));
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 2, 3, 4, 5})
    void testIsPositive(int number) {
        assertTrue(calculator.isPositive(number));
    }

    @ParameterizedTest
    @CsvSource({"1, 2, 3", "5, 5, 10", "0, 0, 0"})
    void testAdditionWithParameters(int a, int b, int expected) {
        assertEquals(expected, calculator.add(a, b));
    }

    @Test
    @DisplayName("Test division by zero")
    void testDivisionByZero() {
        Exception exception = assertThrows(ArithmeticException.class, () -> {
            calculator.divide(10, 0);
        });

        String expectedMessage = "Division by zero";
        String actualMessage = exception.getMessage();
        assertTrue(actualMessage.contains(expectedMessage));
    }
}
```

#### Mockito Testing

```java
import org.mockito.*;
import static org.mockito.Mockito.*;

class UserServiceTest {
    @Mock
    private UserRepository userRepository;

    @Mock
    private EmailService emailService;

    @InjectMocks
    private UserService userService;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testCreateUser() {
        // Arrange
        User user = new User("Alice", "alice@example.com");
        when(userRepository.save(any(User.class))).thenReturn(user);
        doNothing().when(emailService).sendWelcomeEmail(anyString());

        // Act
        User createdUser = userService.createUser("Alice", "alice@example.com");

        // Assert
        assertNotNull(createdUser);
        assertEquals("Alice", createdUser.getName());
        verify(userRepository).save(any(User.class));
        verify(emailService).sendWelcomeEmail("alice@example.com");
    }

    @Test
    void testCreateUserWithInvalidEmail() {
        // Act & Assert
        assertThrows(IllegalArgumentException.class, () -> {
            userService.createUser("Alice", "invalid-email");
        });

        verify(userRepository, never()).save(any(User.class));
        verify(emailService, never()).sendWelcomeEmail(anyString());
    }
}
```

#### Integration Testing

```java
@SpringBootTest
@AutoConfigureTestDatabase(replace = AutoConfigureTestDatabase.Replace.NONE)
class UserServiceIntegrationTest {
    @Autowired
    private UserService userService;

    @Autowired
    private UserRepository userRepository;

    @Test
    void testCreateAndRetrieveUser() {
        // Create user
        User user = userService.createUser("Bob", "bob@example.com");

        // Verify user was saved
        User retrievedUser = userRepository.findById(user.getId()).orElse(null);
        assertNotNull(retrievedUser);
        assertEquals("Bob", retrievedUser.getName());
        assertEquals("bob@example.com", retrievedUser.getEmail());
    }
}
```

### Test Coverage

```java
// Using JaCoCo for code coverage
// Add to build.gradle or pom.xml

// Run tests with coverage
// mvn clean test jacoco:report
// gradle test jacocoTestReport
```

---

## Performance Optimization

### Memory Management

```java
// Object pooling for expensive objects
public class ObjectPool<T> {
    private final Queue<T> pool;
    private final Supplier<T> factory;

    public ObjectPool(int size, Supplier<T> factory) {
        this.pool = new ConcurrentLinkedQueue<>();
        this.factory = factory;

        for (int i = 0; i < size; i++) {
            pool.offer(factory.get());
        }
    }

    public T borrow() {
        T obj = pool.poll();
        return obj != null ? obj : factory.get();
    }

    public void release(T obj) {
        pool.offer(obj);
    }
}

// Usage
ObjectPool<StringBuilder> stringBuilderPool = new ObjectPool<>(10, StringBuilder::new);
StringBuilder sb = stringBuilderPool.borrow();
try {
    sb.append("Hello, World!");
    return sb.toString();
} finally {
    stringBuilderPool.release(sb);
}
```

### String Optimization

```java
public class StringOptimization {
    // Use StringBuilder for multiple concatenations
    public String buildLargeString(List<String> items) {
        StringBuilder sb = new StringBuilder();
        for (String item : items) {
            sb.append(item).append("\n");
        }
        return sb.toString();
    }

    // Use String.format sparingly
    public String formatMessage(String name, int age) {
        return String.format("Name: %s, Age: %d", name, age);
    }

    // Use String.join for simple concatenation
    public String joinNames(List<String> names) {
        return String.join(", ", names);
    }
}
```

### Collection Optimization

```java
public class CollectionOptimization {
    // Pre-size collections when possible
    public List<String> createList(int expectedSize) {
        return new ArrayList<>(expectedSize);
    }

    // Use appropriate collection types
    public void processData() {
        // For frequent lookups
        Map<String, Integer> lookupMap = new HashMap<>();

        // For sorted data
        SortedMap<String, Integer> sortedMap = new TreeMap<>();

        // For thread-safe operations
        Map<String, Integer> concurrentMap = new ConcurrentHashMap<>();

        // For unique elements with fast access
        Set<String> uniqueSet = new HashSet<>();
    }
}
```

### Profiling and Monitoring

```java
import java.lang.management.*;

public class PerformanceMonitor {
    public void monitorMemory() {
        MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
        MemoryUsage heapUsage = memoryBean.getHeapMemoryUsage();

        System.out.println("Used heap: " + heapUsage.getUsed() + " bytes");
        System.out.println("Max heap: " + heapUsage.getMax() + " bytes");
    }

    public void monitorThreads() {
        ThreadMXBean threadBean = ManagementFactory.getThreadMXBean();
        long[] threadIds = threadBean.getAllThreadIds();

        System.out.println("Active threads: " + threadIds.length);
    }
}
```

### JVM Tuning

```bash
# Common JVM options for performance
java -Xms512m -Xmx2g -XX:+UseG1GC -XX:MaxGCPauseMillis=200 MyApplication

# Options explained:
# -Xms512m: Initial heap size
# -Xmx2g: Maximum heap size
# -XX:+UseG1GC: Use G1 garbage collector
# -XX:MaxGCPauseMillis=200: Target maximum GC pause time
```

---

## Java Memory Management

### Garbage Collection

```java
// Understanding garbage collection
public class GarbageCollectionExample {
    public static void main(String[] args) {
        // Create objects
        for (int i = 0; i < 1000000; i++) {
            new Object(); // These will be garbage collected
        }

        // Request garbage collection (not guaranteed)
        System.gc();

        // Get memory info
        Runtime runtime = Runtime.getRuntime();
        long totalMemory = runtime.totalMemory();
        long freeMemory = runtime.freeMemory();
        long usedMemory = totalMemory - freeMemory;

        System.out.println("Used memory: " + usedMemory + " bytes");
    }
}
```

### Memory Leaks

```java
// Common memory leak patterns
public class MemoryLeakExample {
    // Static collection holding references
    private static final List<Object> staticList = new ArrayList<>();

    public void addToStaticList(Object obj) {
        staticList.add(obj); // Memory leak - objects never removed
    }

    // Inner class holding outer reference
    public class InnerClass {
        public void doSomething() {
            // Has implicit reference to outer class
        }
    }

    // Unclosed resources
    public void readFile(String filename) {
        try {
            FileInputStream fis = new FileInputStream(filename);
            // Missing fis.close() - resource leak
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
```

### Weak References

```java
import java.lang.ref.*;

public class WeakReferenceExample {
    public void demonstrateWeakReferences() {
        // Strong reference
        String strongRef = new String("Hello");

        // Weak reference
        WeakReference<String> weakRef = new WeakReference<>(strongRef);

        System.out.println("Weak ref: " + weakRef.get());

        // Remove strong reference
        strongRef = null;

        // Request garbage collection
        System.gc();

        // Weak reference may be null now
        System.out.println("Weak ref after GC: " + weakRef.get());
    }
}
```

---

## Design Patterns

### Creational Patterns

#### Singleton Pattern

```java
public class Singleton {
    private static volatile Singleton instance;
    private Singleton() {}

    public static Singleton getInstance() {
        if (instance == null) {
            synchronized (Singleton.class) {
                if (instance == null) {
                    instance = new Singleton();
                }
            }
        }
        return instance;
    }
}
```

#### Factory Pattern

```java
public interface Animal {
    void makeSound();
}

public class Dog implements Animal {
    public void makeSound() {
        System.out.println("Woof!");
    }
}

public class Cat implements Animal {
    public void makeSound() {
        System.out.println("Meow!");
    }
}

public class AnimalFactory {
    public static Animal createAnimal(String type) {
        switch (type.toLowerCase()) {
            case "dog":
                return new Dog();
            case "cat":
                return new Cat();
            default:
                throw new IllegalArgumentException("Unknown animal type: " + type);
        }
    }
}
```

#### Builder Pattern

```java
public class User {
    private final String name;
    private final String email;
    private final int age;
    private final String address;

    private User(Builder builder) {
        this.name = builder.name;
        this.email = builder.email;
        this.age = builder.age;
        this.address = builder.address;
    }

    public static class Builder {
        private String name;
        private String email;
        private int age;
        private String address;

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder email(String email) {
            this.email = email;
            return this;
        }

        public Builder age(int age) {
            this.age = age;
            return this;
        }

        public Builder address(String address) {
            this.address = address;
            return this;
        }

        public User build() {
            if (name == null || email == null) {
                throw new IllegalStateException("Name and email are required");
            }
            return new User(this);
        }
    }
}

// Usage
User user = new User.Builder()
    .name("Alice")
    .email("alice@example.com")
    .age(25)
    .build();
```

### Structural Patterns

#### Adapter Pattern

```java
// Target interface
public interface PaymentProcessor {
    void processPayment(double amount);
}

// Adaptee (existing class)
public class LegacyPaymentSystem {
    public void pay(double amount, String currency) {
        System.out.println("Paid " + amount + " " + currency);
    }
}

// Adapter
public class PaymentAdapter implements PaymentProcessor {
    private LegacyPaymentSystem legacySystem;

    public PaymentAdapter(LegacyPaymentSystem legacySystem) {
        this.legacySystem = legacySystem;
    }

    @Override
    public void processPayment(double amount) {
        legacySystem.pay(amount, "USD");
    }
}
```

#### Decorator Pattern

```java
public interface Coffee {
    double getCost();
    String getDescription();
}

public class SimpleCoffee implements Coffee {
    public double getCost() { return 1.0; }
    public String getDescription() { return "Simple coffee"; }
}

public abstract class CoffeeDecorator implements Coffee {
    protected Coffee coffee;

    public CoffeeDecorator(Coffee coffee) {
        this.coffee = coffee;
    }

    public double getCost() { return coffee.getCost(); }
    public String getDescription() { return coffee.getDescription(); }
}

public class MilkDecorator extends CoffeeDecorator {
    public MilkDecorator(Coffee coffee) {
        super(coffee);
    }

    public double getCost() { return super.getCost() + 0.5; }
    public String getDescription() { return super.getDescription() + ", milk"; }
}

// Usage
Coffee coffee = new SimpleCoffee();
coffee = new MilkDecorator(coffee);
System.out.println(coffee.getDescription() + ": $" + coffee.getCost());
```

### Behavioral Patterns

#### Observer Pattern

```java
import java.util.*;

public interface Observer {
    void update(String message);
}

public class Subject {
    private List<Observer> observers = new ArrayList<>();

    public void attach(Observer observer) {
        observers.add(observer);
    }

    public void detach(Observer observer) {
        observers.remove(observer);
    }

    public void notifyObservers(String message) {
        for (Observer observer : observers) {
            observer.update(message);
        }
    }
}

public class ConcreteObserver implements Observer {
    private String name;

    public ConcreteObserver(String name) {
        this.name = name;
    }

    @Override
    public void update(String message) {
        System.out.println(name + " received: " + message);
    }
}
```

#### Strategy Pattern

```java
public interface PaymentStrategy {
    void pay(double amount);
}

public class CreditCardPayment implements PaymentStrategy {
    private String cardNumber;

    public CreditCardPayment(String cardNumber) {
        this.cardNumber = cardNumber;
    }

    @Override
    public void pay(double amount) {
        System.out.println("Paid " + amount + " using credit card");
    }
}

public class PayPalPayment implements PaymentStrategy {
    private String email;

    public PayPalPayment(String email) {
        this.email = email;
    }

    @Override
    public void pay(double amount) {
        System.out.println("Paid " + amount + " using PayPal");
    }
}

public class ShoppingCart {
    private PaymentStrategy paymentStrategy;

    public void setPaymentStrategy(PaymentStrategy strategy) {
        this.paymentStrategy = strategy;
    }

    public void checkout(double amount) {
        paymentStrategy.pay(amount);
    }
}
```

---

## Conclusion

This comprehensive guide covers all the essential aspects of Java programming, from basic syntax to advanced concepts. Here's a summary of what we've covered:

### Core Concepts

- **Language Fundamentals**: Syntax, data types, control flow, methods
- **Object-Oriented Programming**: Classes, inheritance, interfaces, polymorphism
- **Collections Framework**: Lists, Sets, Maps, and their implementations
- **Exception Handling**: Try-catch blocks, custom exceptions, resource management
- **Generics**: Type-safe collections and methods

### Advanced Topics

- **Lambda Expressions and Streams**: Functional programming in Java
- **Annotations**: Metadata and compile-time processing
- **Reflection**: Runtime class inspection and manipulation
- **Serialization**: Object persistence and transmission
- **Networking**: HTTP clients, socket programming
- **Database Connectivity**: JDBC and connection pooling

### Best Practices

- **Code Style**: Naming conventions, documentation, immutability
- **Performance**: Memory management, collection optimization, profiling
- **Testing**: Unit testing, mocking, integration testing
- **Design Patterns**: Creational, structural, and behavioral patterns

### Development Tools

- **Build Tools**: Maven and Gradle
- **IDEs**: IntelliJ IDEA, Eclipse, NetBeans
- **Libraries**: Core Java APIs and popular third-party libraries
- **Debugging**: IDE debugging, logging, profiling

### Key Takeaways

1. **Practice Regularly**: Java is a vast language - continuous learning is essential
2. **Follow Standards**: Adhere to Java coding conventions and best practices
3. **Write Tests**: Comprehensive testing ensures code quality and reliability
4. **Stay Updated**: Java evolves rapidly - keep up with new features and libraries
5. **Understand the JVM**: Knowledge of memory management and garbage collection is crucial
6. **Use Design Patterns**: They provide proven solutions to common problems
7. **Profile Your Code**: Performance optimization requires measurement and analysis

### Learning Path

1. **Beginner**: Start with basic syntax, OOP concepts, and collections
2. **Intermediate**: Learn generics, exceptions, file I/O, and basic threading
3. **Advanced**: Master streams, reflection, design patterns, and performance optimization
4. **Expert**: Deep dive into JVM internals, advanced concurrency, and enterprise patterns

### Additional Resources

#### Official Documentation

- [Java SE Documentation](https://docs.oracle.com/en/java/)
- [Java Tutorials](https://docs.oracle.com/javase/tutorial/)
- [Java API Documentation](https://docs.oracle.com/en/java/javase/)

#### Books

- **Effective Java** by Joshua Bloch
- **Clean Code** by Robert C. Martin
- **Java Concurrency in Practice** by Brian Goetz
- **Design Patterns** by Gang of Four
- **Java Performance** by Scott Oaks

#### Online Resources

- [Baeldung Java Tutorials](https://www.baeldung.com/)
- [Java Code Geeks](https://www.javacodegeeks.com/)
- [DZone Java](https://dzone.com/java)
- [Stack Overflow Java](https://stackoverflow.com/questions/tagged/java)

#### Communities

- [Java Reddit](https://www.reddit.com/r/java/)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/java)
- [Java User Groups](https://www.oracle.com/java/technologies/jug.html)

#### Tools and Frameworks

- **Build Tools**: Maven, Gradle
- **Testing**: JUnit, TestNG, Mockito
- **Logging**: SLF4J, Log4j2, Logback
- **JSON Processing**: Jackson, Gson
- **HTTP Clients**: Apache HttpClient, OkHttp
- **Database**: Hibernate, MyBatis, Spring Data

### Final Thoughts

Java remains one of the most popular and versatile programming languages in the world. Its strong typing, object-oriented nature, and extensive ecosystem make it ideal for building robust, scalable applications. Whether you're developing web applications, mobile apps, enterprise software, or embedded systems, Java provides the tools and frameworks you need.

Remember that becoming proficient in Java is a journey, not a destination. The language and ecosystem continue to evolve, so stay curious, keep learning, and never stop practicing.

Happy coding in Java! ☕

---

_This guide is a living document. As Java evolves, so should your knowledge. Keep exploring, experimenting, and building amazing applications with Java!_
