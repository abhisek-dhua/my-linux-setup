# Angular Interview Questions & Answers

## Table of Contents

1. [Core Angular Concepts](#core-angular-concepts)
2. [Component Lifecycle & Change Detection](#component-lifecycle--change-detection)
3. [Dependency Injection](#dependency-injection)
4. [Angular Modules & Architecture](#angular-modules--architecture)
5. [Data Binding & Templates](#data-binding--templates)
6. [Directives & Pipes](#directives--pipes)
7. [Routing & Navigation](#routing--navigation)
8. [Services & HTTP](#services--http)
9. [Forms & Validation](#forms--validation)
10. [Performance Optimization](#performance-optimization)

---

## 1. Core Angular Concepts

### Q1: What is Angular and how does it differ from AngularJS?

**Answer:**
Angular is a TypeScript-based, component-based framework for building scalable web applications. It was completely rewritten from AngularJS (version 1.x) and offers significant improvements:

**Key Differences:**

- **Language**: Angular uses TypeScript (superset of JavaScript) while AngularJS uses JavaScript
- **Architecture**: Angular uses component-based architecture vs AngularJS's controller-based architecture
- **Performance**: Angular has better performance due to improved change detection and rendering
- **Mobile Support**: Angular has better mobile support compared to AngularJS
- **Dependency Injection**: Angular has a more advanced and hierarchical DI system
- **Templates**: Angular uses a more powerful template syntax with better type checking

**Angular Features:**

- Component-based architecture
- TypeScript support for better development experience
- Improved dependency injection
- Enhanced routing capabilities
- Better performance through change detection optimization
- CLI tools for project scaffolding and management

### Q2: Explain the component architecture in Angular

**Answer:**
Angular's component architecture is a hierarchical structure where components form a tree-like organization:

**Component Structure:**

```typescript
@Component({
  selector: 'app-user-profile',
  templateUrl: './user-profile.component.html',
  styleUrls: ['./user-profile.component.css'],
})
export class UserProfileComponent {
  @Input() user: User;
  @Output() userUpdated = new EventEmitter<User>();

  constructor(private userService: UserService) {}

  ngOnInit() {
    // Component initialization logic
  }
}
```

**Key Concepts:**

- **Components**: Reusable UI building blocks with their own template, logic, and style
- **Templates**: HTML with Angular-specific syntax for data binding and directives
- **Metadata**: Decorators that provide configuration information
- **Component Tree**: Parent-child relationships forming a hierarchical structure
- **Encapsulation**: Each component encapsulates its own logic, template, and styles

**Component Lifecycle:**

1. Creation and initialization
2. Data binding and rendering
3. Updates and changes
4. Destruction and cleanup

### Q3: What are Angular decorators and what are the main types?

**Answer:**
Decorators are functions that modify classes, methods, or properties. Angular uses decorators extensively to add metadata and functionality:

**Main Decorator Types:**

1. **@Component**: Defines a component with template, styles, and selector

```typescript
@Component({
  selector: 'app-example',
  templateUrl: './example.component.html',
  styleUrls: ['./example.component.css']
})
```

2. **@Directive**: Creates custom directives for DOM manipulation

```typescript
@Directive({
  selector: '[appHighlight]',
})
export class HighlightDirective {}
```

3. **@Pipe**: Creates custom pipes for data transformation

```typescript
@Pipe({
  name: 'customPipe',
})
export class CustomPipe implements PipeTransform {}
```

4. **@Injectable**: Marks a class as available for dependency injection

```typescript
@Injectable({
  providedIn: 'root',
})
export class DataService {}
```

5. **@Input/@Output**: Defines component inputs and outputs for communication

```typescript
@Input() data: any;
@Output() dataChange = new EventEmitter<any>();
```

**Purpose of Decorators:**

- Provide metadata for Angular's compilation process
- Enable dependency injection
- Define component behavior and configuration
- Facilitate template binding and event handling

### Q4: What is the difference between a component and a directive in Angular?

**Answer:**

**Components:**

- Extend directives with template functionality
- Have their own template, styles, and view
- Create new views and DOM elements
- Use @Component decorator
- Can contain other components and directives
- Have a lifecycle with hooks (ngOnInit, ngOnDestroy, etc.)
- Are the building blocks of Angular applications

**Directives:**

- Modify existing DOM elements
- Do not have templates
- Enhance or change the behavior of existing elements
- Use @Directive decorator
- Can be structural (change DOM layout) or attribute (change element appearance/behavior)
- Do not have component lifecycle hooks
- Are used for DOM manipulation and behavior enhancement

**Example Usage:**

```typescript
// Component - creates new view
@Component({
  selector: 'app-user-card',
  template: '<div>User: {{name}}</div>',
})
export class UserCardComponent {}

// Directive - modifies existing element
@Directive({
  selector: '[appHighlight]',
})
export class HighlightDirective {
  constructor(private el: ElementRef) {
    el.nativeElement.style.backgroundColor = 'yellow';
  }
}
```

### Q5: Explain Angular's module system (NgModule)

**Answer:**
Angular modules (NgModules) are containers that organize related components, directives, pipes, and services:

**NgModule Structure:**

```typescript
@NgModule({
  declarations: [
    // Components, directives, and pipes that belong to this module
    MyComponent,
    MyDirective,
    MyPipe,
  ],
  imports: [
    // Other modules whose classes are needed by this module
    CommonModule,
    RouterModule,
    FormsModule,
  ],
  exports: [
    // Components, directives, and pipes visible to other modules
    MyComponent,
    MyPipe,
  ],
  providers: [
    // Services available to this module
    MyService,
  ],
  bootstrap: [
    // Root component to bootstrap
    AppComponent,
  ],
})
export class AppModule {}
```

**Types of Modules:**

1. **Root Module**: Bootstraps the application (typically AppModule)
2. **Feature Modules**: Organize related functionality
3. **Shared Modules**: Reusable components, directives, and pipes
4. **Core Modules**: Singleton services and application-wide components
5. **Routing Modules**: Route configurations

**Module Benefits:**

- Code organization and modularity
- Dependency management
- Lazy loading capabilities
- Reusability across applications
- Clear separation of concerns

---

## 2. Component Lifecycle & Change Detection

### Q6: Explain Angular's component lifecycle hooks

**Answer:**
Angular components go through a lifecycle with specific hooks that allow you to tap into different phases:

**Lifecycle Hooks in Order:**

1. **ngOnChanges**: Called when @Input properties change

```typescript
ngOnChanges(changes: SimpleChanges) {
  // React to input property changes
  if (changes['userId']) {
    this.loadUser(changes['userId'].currentValue);
  }
}
```

2. **ngOnInit**: Called once after first ngOnChanges, for initialization

```typescript
ngOnInit() {
  // Initialize component, fetch data, set up subscriptions
  this.loadInitialData();
}
```

3. **ngDoCheck**: Called during every change detection cycle

```typescript
ngDoCheck() {
  // Custom change detection logic
  // Use with caution as it runs frequently
}
```

4. **ngAfterContentInit**: Called after content (ng-content) is projected

```typescript
ngAfterContentInit() {
  // Access projected content
}
```

5. **ngAfterContentChecked**: Called after projected content is checked

```typescript
ngAfterContentChecked() {
  // React to changes in projected content
}
```

6. **ngAfterViewInit**: Called after component's view is initialized

```typescript
ngAfterViewInit() {
  // Access child components and DOM elements
  this.childComponent.someMethod();
}
```

7. **ngAfterViewChecked**: Called after component's view is checked

```typescript
ngAfterViewChecked() {
  // React to changes in component's view
}
```

8. **ngOnDestroy**: Called just before component is destroyed

```typescript
ngOnDestroy() {
  // Cleanup: unsubscribe from observables, remove event listeners
  this.subscription.unsubscribe();
}
```

**Usage Guidelines:**

- Use ngOnInit for initialization logic
- Use ngOnDestroy for cleanup to prevent memory leaks
- Avoid heavy operations in ngDoCheck and ngAfterViewChecked
- Use ngAfterViewInit for DOM manipulation and child component access

### Q7: How does Angular's change detection work?

**Answer:**
Angular's change detection is a mechanism that checks for changes in component data and updates the DOM accordingly:

**Change Detection Process:**

1. **Zone.js Integration**: Angular uses Zone.js to intercept asynchronous operations
2. **Dirty Checking**: After async operations, Angular runs change detection
3. **Tree Traversal**: Checks all components from root to leaves
4. **Value Comparison**: Compares current and previous values
5. **DOM Updates**: Updates DOM if changes are detected

**Change Detection Strategies:**

1. **Default Strategy** (DefaultChangeDetector):

```typescript
@Component({
  selector: 'app-default',
  changeDetection: ChangeDetectionStrategy.Default,
})
export class DefaultComponent {
  // Change detection runs on every async operation
}
```

2. **OnPush Strategy** (OnPushChangeDetector):

```typescript
@Component({
  selector: 'app-onpush',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class OnPushComponent {
  @Input() data: any;

  // Change detection only runs when:
  // - @Input references change
  // - Events originate from component or children
  // - Async pipes receive new values
}
```

**Performance Optimization with OnPush:**

- Reduces unnecessary change detection cycles
- Improves application performance
- Requires immutable data patterns
- Components only check when inputs change

**Manual Change Detection:**

```typescript
import { ChangeDetectorRef } from '@angular/core';

export class MyComponent {
  constructor(private cd: ChangeDetectorRef) {}

  updateData() {
    this.data = newData;
    this.cd.detectChanges(); // Force change detection
  }
}
```

### Q8: What is the difference between constructor and ngOnInit?

**Answer:**

**Constructor:**

- Part of ES6 class syntax, not Angular-specific
- Called during class instantiation
- Used for dependency injection and basic initialization
- Should be kept minimal and fast
- Runs before Angular has set up the component

```typescript
export class MyComponent {
  constructor(
    private service: MyService,
    private route: ActivatedRoute,
  ) {
    // Dependency injection only
    // Avoid complex logic here
  }
}
```

**ngOnInit:**

- Angular lifecycle hook
- Called after Angular has initialized all data-bound properties
- Called once after the first ngOnChanges
- Ideal for component initialization logic
- Safe to access @Input properties and services

```typescript
export class MyComponent implements OnInit {
  @Input() userId: string;

  constructor(private service: MyService) {}

  ngOnInit() {
    // Safe to access @Input properties
    if (this.userId) {
      this.loadUserData(this.userId);
    }

    // Initialize subscriptions
    this.setupSubscriptions();

    // Fetch initial data
    this.service.getData().subscribe((data) => {
      this.data = data;
    });
  }
}
```

**Best Practices:**

- Use constructor for dependency injection only
- Use ngOnInit for all initialization logic
- Keep constructor lightweight
- Access @Input properties in ngOnInit, not constructor
- Perform async operations in ngOnInit

### Q9: How do you handle component communication in Angular?

**Answer:**
Angular provides several mechanisms for component communication:

**1. Parent to Child (@Input):**

```typescript
// Parent Component
<app-child [data]="parentData" [config]="config"></app-child>

// Child Component
@Input() data: any;
@Input() config: Config;
```

**2. Child to Parent (@Output + EventEmitter):**

```typescript
// Child Component
@Output() dataChanged = new EventEmitter<any>();

updateData() {
  this.dataChanged.emit(this.data);
}

// Parent Template
<app-child (dataChanged)="handleDataChange($event)"></app-child>
```

**3. Sibling Communication (Services):**

```typescript
@Injectable({
  providedIn: 'root',
})
export class DataService {
  private dataSubject = new BehaviorSubject<any>(null);
  data$ = this.dataSubject.asObservable();

  updateData(data: any) {
    this.dataSubject.next(data);
  }
}

// Component A
this.dataService.updateData(newData);

// Component B
this.dataService.data$.subscribe((data) => {
  this.data = data;
});
```

**4. ViewChild/ViewChildren:**

```typescript
// Parent Component
@ViewChild(ChildComponent) child: ChildComponent;

callChildMethod() {
  this.child.someMethod();
}

// Child Component
@ViewChildren('.item') items: QueryList<ElementRef>;
```

**5. Router State:**

```typescript
// Navigate with state
this.router.navigate(['/detail'], {
  state: { data: this.data },
});

// Access state in target component
const state = history.state;
```

**6. LocalStorage/SessionStorage:**

```typescript
// Store data
localStorage.setItem('userData', JSON.stringify(data));

// Retrieve data
const data = JSON.parse(localStorage.getItem('userData'));
```

**Communication Strategy Selection:**

- Use @Input/@Output for direct parent-child communication
- Use services for complex communication or when components aren't directly related
- Use ViewChild for parent to access child methods/properties
- Use router state for navigation-based data passing
- Use storage for persistence across sessions

### Q10: What are Angular pipes and how do you create custom pipes?

**Answer:**
Pipes are simple functions that accept input and return transformed output, used primarily in templates for data transformation:

**Built-in Pipes:**

```typescript
{{ value | uppercase }}
{{ date | date:'short' }}
{{ price | currency:'USD':'symbol':'1.2-2' }}
{{ text | slice:0:10 }}
{{ object | json }}
```

**Creating Custom Pipes:**

1. **Pure Pipe** (Default - only recalculates when input changes):

```typescript
@Pipe({
  name: 'customFormat'
})
export class CustomFormatPipe implements PipeTransform {
  transform(value: any, ...args: any[]): any {
    // Transform logic here
    return transformedValue;
  }
}

// Usage in template
{{ data | customFormat:'arg1':'arg2' }}
```

2. **Impure Pipe** (Recalculates on every change detection):

```typescript
@Pipe({
  name: 'searchFilter',
  pure: false,
})
export class SearchFilterPipe implements PipeTransform {
  transform(items: any[], searchTerm: string): any[] {
    if (!items || !searchTerm) {
      return items;
    }

    return items.filter((item) => item.name.toLowerCase().includes(searchTerm.toLowerCase()));
  }
}
```

**Advanced Custom Pipe Example:**

```typescript
@Pipe({
  name: 'numberAbbreviation'
})
export class NumberAbbreviationPipe implements PipeTransform {
  transform(value: number, decimals: number = 2): string {
    if (value === null || value === undefined) return '';

    const absValue = Math.abs(value);
    const sign = value < 0 ? '-' : '';

    if (absValue >= 1e9) {
      return `${sign}${(absValue / 1e9).toFixed(decimals)}B`;
    } else if (absValue >= 1e6) {
      return `${sign}${(absValue / 1e6).toFixed(decimals)}M`;
    } else if (absValue >= 1e3) {
      return `${sign}${(absValue / 1e3).toFixed(decimals)}K`;
    }

    return `${sign}${absValue}`;
  }
}

// Usage
{{ 1500000 | numberAbbreviation }} // "1.50M"
{{ 1234567890 | numberAbbreviation:1 }} // "1.2B"
```

**Pipe Best Practices:**

- Keep pipes pure when possible for performance
- Use pipes for presentation logic only
- Avoid complex calculations in pipes
- Consider memoization for expensive operations
- Register pipes in the appropriate module

**Pipe Categories:**

- **Pure Pipes**: Only re-evaluate when input changes (recommended)
- **Impure Pipes**: Re-evaluate on every change detection cycle
- **Async Pipes**: Handle Observable and Promise values automatically

---

## 3. Dependency Injection

### Q11: What is Dependency Injection in Angular and how does it work?

**Answer:**
Dependency Injection (DI) is a design pattern where a class receives its dependencies from external sources rather than creating them itself. Angular has a built-in DI system that manages how components, services, and other classes get their dependencies.

**How Angular DI Works:**

1. **Injector**: Creates and manages dependencies
2. **Provider**: Tells the injector how to create a dependency
3. **Token**: Identifies a dependency for the injector
4. **Dependency**: The service or value that gets injected

**DI Architecture:**

```typescript
// Service definition
@Injectable({
  providedIn: 'root', // Root injector provides this service
})
export class UserService {
  getUsers() {
    return this.http.get('/api/users');
  }
}

// Component using DI
@Component({
  selector: 'app-user-list',
  template: '<div>User list component</div>',
})
export class UserListComponent {
  constructor(private userService: UserService) {
    // Angular injects UserService instance
  }
}
```

**Provider Types:**

1. **Class Provider**:

```typescript
providers: [UserService];
// Equivalent to: { provide: UserService, useClass: UserService }
```

2. **Value Provider**:

```typescript
providers: [{ provide: 'API_URL', useValue: 'https://api.example.com' }];
```

3. **Factory Provider**:

```typescript
providers: [
  {
    provide: 'CONFIG',
    useFactory: (env: EnvironmentService) => {
      return env.isProduction ? prodConfig : devConfig;
    },
    deps: [EnvironmentService],
  },
];
```

4. **Alias Provider**:

```typescript
providers: [
  { provide: LoggerService, useClass: ConsoleLoggerService },
  { provide: SpecialLoggerService, useExisting: LoggerService },
];
```

**Injector Hierarchy:**

- **Root Injector**: Created by @Injectable({ providedIn: 'root' })
- **Module Injectors**: Created by NgModule providers
- **Component Injectors**: Created by Component providers
- **Element Injectors**: Created by Directive providers

**DI Benefits:**

- Loose coupling between components and services
- Easier testing with mock dependencies
- Better code organization and reusability
- Automatic lifecycle management

### Q12: Explain hierarchical injectors in Angular

**Answer:**
Angular's hierarchical injector system creates a tree of injectors that mirrors the component tree, allowing for different levels of dependency resolution:

**Injector Hierarchy Levels:**

1. **Null Injector**: Root of the hierarchy, throws error if dependency not found
2. **Platform Injector**: For platform-wide services (BrowserModule, ServiceWorker)
3. **Root Injector**: Created by @Injectable({ providedIn: 'root' }) or NgModule providers
4. **Module Injectors**: Created by feature modules
5. **Component Injectors**: Created by Component providers array
6. **Element Injectors**: Created by Directive providers

**Resolution Strategy:**
When a component requests a dependency, Angular searches up the injector hierarchy:

```typescript
@Component({
  selector: 'app-root',
  providers: [RootService], // Available to AppComponent and children
})
export class AppComponent {}

@Component({
  selector: 'app-child',
  providers: [ChildService], // Available only to ChildComponent and its children
})
export class ChildComponent {
  constructor(
    private rootService: RootService, // Found in parent injector
    private childService: ChildService, // Found in local injector
  ) {}
}
```

**Injector Creation Examples:**

1. **Root Level**:

```typescript
@Injectable({
  providedIn: 'root', // Creates provider in root injector
})
export class GlobalService {}
```

2. **Module Level**:

```typescript
@NgModule({
  providers: [ModuleService], // Available throughout module
})
export class FeatureModule {}
```

3. **Component Level**:

```typescript
@Component({
  providers: [LocalService], // Only available to this component and children
})
export class MyComponent {}
```

**Injector Scope Rules:**

- Child injectors can see parent dependencies
- Parent injectors cannot see child dependencies
- Multiple components can share the same dependency instance
- Each injector maintains its own instance of provided dependencies

**Use Cases for Different Levels:**

- **Root**: Application-wide services (HTTP, authentication)
- **Module**: Feature-specific services
- **Component**: Component-specific services or different instances

### Q13: How do you create and provide services in Angular?

**Answer:**
Services in Angular are classes that handle business logic, data management, and external communications:

**Creating Services:**

1. **Basic Service**:

```typescript
@Injectable({
  providedIn: 'root',
})
export class DataService {
  private dataSubject = new BehaviorSubject<any[]>([]);
  data$ = this.dataSubject.asObservable();

  constructor(private http: HttpClient) {}

  fetchData(): Observable<any[]> {
    return this.http.get<any[]>('/api/data').pipe(tap((data) => this.dataSubject.next(data)));
  }

  updateData(data: any[]): void {
    this.dataSubject.next(data);
  }
}
```

2. **Service with Dependencies**:

```typescript
@Injectable({
  providedIn: 'root',
})
export class AuthService {
  private currentUserSubject = new BehaviorSubject<User | null>(null);
  currentUser$ = this.currentUserSubject.asObservable();

  constructor(
    private http: HttpClient,
    private router: Router,
    @Inject('API_URL') private apiUrl: string,
  ) {}

  login(credentials: LoginCredentials): Observable<User> {
    return this.http.post<User>(`${this.apiUrl}/auth/login`, credentials).pipe(
      tap((user) => {
        this.currentUserSubject.next(user);
        localStorage.setItem('token', user.token);
      }),
    );
  }
}
```

**Service Provision Methods:**

1. **Root Level (Recommended)**:

```typescript
@Injectable({
  providedIn: 'root',
})
export class UserService {}
```

2. **Module Level**:

```typescript
@NgModule({
  providers: [UserService],
})
export class UserModule {}
```

3. **Component Level**:

```typescript
@Component({
  providers: [UserService],
})
export class UserComponent {}
```

**Service Patterns:**

1. **Singleton Pattern** (Default):

```typescript
@Injectable({
  providedIn: 'root',
})
export class SingletonService {
  // Same instance shared across application
}
```

2. **Multiple Instances**:

```typescript
@Injectable()
export class UniqueService {
  private id = Math.random();

  getId() {
    return this.id;
  }
}

// Each component gets different instance
@Component({
  providers: [UniqueService],
})
export class ComponentA {}

@Component({
  providers: [UniqueService],
})
export class ComponentB {}
```

3. **Factory Pattern**:

```typescript
@Injectable()
export class ConfigService {
  constructor(@Inject('CONFIG') private config: AppConfig) {}
}

export function configFactory(): AppConfig {
  return {
    apiUrl: environment.apiUrl,
    timeout: 5000,
  };
}

// In module
providers: [
  {
    provide: 'CONFIG',
    useFactory: configFactory,
  },
  ConfigService,
];
```

**Service Best Practices:**

- Use providedIn: 'root' for most services
- Keep services focused on specific responsibilities
- Use RxJS for async operations and state management
- Implement proper error handling
- Use dependency injection for testability

### Q14: What are Angular tokens and when would you use them?

**Answer:**
Angular tokens are identifiers used to register and retrieve dependencies in the dependency injection system. They provide a way to inject values that aren't classes.

**Token Types:**

1. **OpaqueToken (Deprecated)**:

```typescript
// Old way (deprecated)
const CONFIG_TOKEN = new OpaqueToken('app.config');
```

2. **InjectionToken (Current)**:

```typescript
const CONFIG_TOKEN = new InjectionToken<AppConfig>('app.config');

// Provider
providers: [
  { provide: CONFIG_TOKEN, useValue: { apiUrl: 'https://api.example.com' } }
]

// Injection
constructor(@Inject(CONFIG_TOKEN) private config: AppConfig) {}
```

**Common Use Cases:**

1. **Configuration Values**:

```typescript
const APP_CONFIG = new InjectionToken<AppConfig>('app.config');

providers: [
  {
    provide: APP_CONFIG,
    useValue: {
      apiUrl: environment.apiUrl,
      timeout: 5000,
      retryCount: 3
    }
  }
]

// Usage
constructor(@Inject(APP_CONFIG) private config: AppConfig) {
  console.log(this.config.apiUrl);
}
```

2. **Feature Flags**:

```typescript
const FEATURE_FLAGS = new InjectionToken<FeatureFlags>('feature.flags');

providers: [
  {
    provide: FEATURE_FLAGS,
    useValue: {
      newDashboard: true,
      darkMode: false,
      premiumFeatures: true,
    },
  },
];
```

3. **Third-party Libraries**:

```typescript
const LOGGER_TOKEN = new InjectionToken<Logger>('logger');

providers: [
  {
    provide: LOGGER_TOKEN,
    useFactory: () => {
      return new WinstonLogger({
        level: 'info',
        format: winston.format.json(),
      });
    },
  },
];
```

4. **Multi Providers**:

```typescript
const VALIDATORS = new InjectionToken<ValidatorFn[]>('validators');

providers: [
  {
    provide: VALIDATORS,
    useValue: [emailValidator, passwordValidator],
    multi: true
  },
  {
    provide: VALIDATORS,
    useValue: [usernameValidator],
    multi: true
  }
]

// Injection
constructor(@Inject(VALIDATORS) private validators: ValidatorFn[]) {
  // validators = [emailValidator, passwordValidator, usernameValidator]
}
```

**Advanced Token Usage:**

1. **Conditional Providers**:

```typescript
const LOG_LEVEL = new InjectionToken<string>('log.level');

providers: [
  {
    provide: LOG_LEVEL,
    useFactory: (env: EnvironmentService) => {
      return env.isProduction() ? 'error' : 'debug';
    },
    deps: [EnvironmentService],
  },
];
```

2. **Async Tokens**:

```typescript
const USER_CONFIG = new InjectionToken<UserConfig>('user.config');

providers: [
  {
    provide: USER_CONFIG,
    useFactory: async (authService: AuthService) => {
      const user = await authService.getCurrentUser().toPromise();
      return user.preferences;
    },
    deps: [AuthService],
  },
];
```

**Token Benefits:**

- Type safety with InjectionToken
- Configuration injection without creating classes
- Multi-provider support for arrays of dependencies
- Conditional and async provider resolution
- Better testing with mock token values

### Q15: How do you handle circular dependencies in Angular?

**Answer:**
Circular dependencies occur when two or more modules depend on each other, creating a dependency loop. Angular provides several strategies to resolve them:

**Common Circular Dependency Scenarios:**

1. **Service A depends on Service B, Service B depends on Service A**:

```typescript
// service-a.service.ts
@Injectable()
export class ServiceA {
  constructor(private serviceB: ServiceB) {}
}

// service-b.service.ts
@Injectable()
export class ServiceB {
  constructor(private serviceA: ServiceA) {}
}
```

**Resolution Strategies:**

1. **Refactor into Shared Service**:

```typescript
// shared.service.ts
@Injectable()
export class SharedService {
  // Common functionality
}

// service-a.service.ts
@Injectable()
export class ServiceA {
  constructor(private sharedService: SharedService) {}
}

// service-b.service.ts
@Injectable()
export class ServiceB {
  constructor(private sharedService: SharedService) {}
}
```

2. **Use Dependency Injection with @Optional()**:

```typescript
@Injectable()
export class ServiceA {
  constructor(@Optional() private serviceB?: ServiceB) {
    // Handle case where serviceB is undefined
  }
}
```

3. **Lazy Loading with forwardRef()**:

```typescript
import { forwardRef } from '@angular/core';

@Injectable()
export class ServiceA {
  constructor(@Inject(forwardRef(() => ServiceB)) private serviceB: ServiceB) {}
}
```

4. **Interface-based Dependencies**:

```typescript
// interfaces.ts
export interface IServiceB {
  doSomething(): void;
}

// service-b.service.ts
@Injectable()
export class ServiceB implements IServiceB {
  doSomething() {
    // Implementation
  }
}

// service-a.service.ts
@Injectable()
export class ServiceA {
  constructor(@Inject('IServiceB') private serviceB: IServiceB) {}
}

// providers
providers: [ServiceB, { provide: 'IServiceB', useExisting: ServiceB }];
```

5. **Event-driven Communication**:

```typescript
@Injectable()
export class ServiceA {
  private eventsSubject = new Subject<any>();
  events$ = this.eventsSubject.asObservable();

  triggerEvent(data: any) {
    this.eventsSubject.next(data);
  }
}

@Injectable()
export class ServiceB {
  constructor(private serviceA: ServiceA) {
    this.serviceA.events$.subscribe((event) => {
      // Handle events from ServiceA
    });
  }
}
```

6. **Module-level Resolution**:

```typescript
// shared.module.ts
@NgModule({
  providers: [ServiceA, ServiceB],
})
export class SharedModule {}

// In feature modules, import SharedModule instead of individual services
```

**Prevention Strategies:**

1. **Follow SOLID Principles**:

```typescript
// Single Responsibility: Each service has one reason to change
// Dependency Inversion: Depend on abstractions, not concretions
```

2. **Use Facade Pattern**:

```typescript
@Injectable()
export class UserFacade {
  constructor(
    private userService: UserService,
    private authService: AuthService,
    private notificationService: NotificationService,
  ) {}

  // Expose only necessary methods
  login(credentials: Credentials) {
    return this.authService
      .login(credentials)
      .pipe(tap((user) => this.notificationService.show('Welcome back!')));
  }
}
```

3. **Proper Module Organization**:

```typescript
// shared/
//   services/
//     user.service.ts
//     auth.service.ts
//     notification.service.ts
//   facades/
//     user.facade.ts
//   models/
//     user.model.ts
```

**Best Practices:**

- Use interfaces for dependencies when possible
- Implement the Facade pattern for complex service interactions
- Organize services by feature, not by type
- Use event-driven architecture for loose coupling
- Regularly review and refactor dependency graphs

---

## 4. Angular Modules & Architecture

### Q16: What are feature modules and how do you create them?

**Answer:**
Feature modules are NgModules that organize related components, services, and functionality into cohesive, reusable units within an Angular application.

**Purpose of Feature Modules:**

- Code organization and modularity
- Lazy loading capabilities
- Reusability across applications
- Clear separation of concerns
- Better maintainability

**Creating a Feature Module:**

```typescript
// user.module.ts
import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';

// Components
import { UserListComponent } from './components/user-list/user-list.component';
import { UserDetailComponent } from './components/user-detail/user-detail.component';
import { UserFormComponent } from './components/user-form/user-form.component';

// Services
import { UserService } from './services/user.service';
import { UserResolver } from './resolvers/user.resolver';

// Guards
import { UserGuard } from './guards/user.guard';

// Pipes
import { UserStatusPipe } from './pipes/user-status.pipe';

// Routing
import { UserRoutingModule } from './user-routing.module';

@NgModule({
  declarations: [UserListComponent, UserDetailComponent, UserFormComponent, UserStatusPipe],
  imports: [CommonModule, FormsModule, ReactiveFormsModule, UserRoutingModule],
  exports: [UserListComponent, UserFormComponent],
  providers: [UserService, UserResolver, UserGuard],
})
export class UserModule {}
```

**Feature Module Structure:**

```
user/
├── components/
│   ├── user-list/
│   │   ├── user-list.component.ts
│   │   ├── user-list.component.html
│   │   └── user-list.component.css
│   ├── user-detail/
│   └── user-form/
├── services/
│   └── user.service.ts
├── guards/
│   └── user.guard.ts
├── resolvers/
│   └── user.resolver.ts
├── pipes/
│   └── user-status.pipe.ts
├── models/
│   └── user.model.ts
├── user.module.ts
└── user-routing.module.ts
```

**Types of Feature Modules:**

1. **Domain Feature Module**:

```typescript
// dashboard.module.ts
@NgModule({
  imports: [CommonModule, DashboardRoutingModule, SharedModule],
  declarations: [DashboardComponent, WidgetComponent],
})
export class DashboardModule {}
```

2. **Routed Feature Module**:

```typescript
// products.module.ts
const routes: Routes = [
  { path: 'products', component: ProductListComponent },
  { path: 'products/:id', component: ProductDetailComponent },
];

@NgModule({
  imports: [RouterModule.forChild(routes), ProductsRoutingModule],
  declarations: [ProductListComponent, ProductDetailComponent],
})
export class ProductsModule {}
```

3. **Routing Module**:

```typescript
// user-routing.module.ts
const routes: Routes = [
  {
    path: 'users',
    component: UserListComponent,
    canActivate: [AuthGuard],
    children: [{ path: ':id', component: UserDetailComponent, resolve: { user: UserResolver } }],
  },
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule],
})
export class UserRoutingModule {}
```

**Lazy Loading Feature Modules:**

```typescript
// app-routing.module.ts
const routes: Routes = [
  { path: 'users', loadChildren: () => import('./user/user.module').then((m) => m.UserModule) },
  {
    path: 'products',
    loadChildren: () => import('./products/products.module').then((m) => m.ProductsModule),
  },
  {
    path: 'dashboard',
    loadChildren: () => import('./dashboard/dashboard.module').then((m) => m.DashboardModule),
  },
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule],
})
export class AppRoutingModule {}
```

**Shared Modules:**

```typescript
// shared.module.ts
@NgModule({
  imports: [CommonModule, FormsModule, ReactiveFormsModule, RouterModule],
  declarations: [LoadingComponent, ErrorComponent, ConfirmDialogComponent],
  exports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,
    RouterModule,
    LoadingComponent,
    ErrorComponent,
    ConfirmDialogComponent,
  ],
})
export class SharedModule {}
```

**Core Module:**

```typescript
// core.module.ts
@NgModule({
  imports: [HttpClientModule, SharedModule],
  providers: [
    { provide: HTTP_INTERCEPTORS, useClass: AuthInterceptor, multi: true },
    { provide: HTTP_INTERCEPTORS, useClass: ErrorInterceptor, multi: true },
  ],
})
export class CoreModule {
  // Prevent re-importing the CoreModule
  constructor(@Optional() @SkipSelf() parentModule: CoreModule) {
    if (parentModule) {
      throw new Error('CoreModule is already loaded. Import it in the AppModule only');
    }
  }
}
```

**Best Practices:**

- Create feature modules for each major feature area
- Use shared modules for reusable components
- Implement core modules for application-wide services
- Use lazy loading for better performance
- Follow consistent naming conventions

### Q17: Explain lazy loading in Angular and its benefits

**Answer:**
Lazy loading is a technique where Angular loads feature modules only when they are needed, rather than loading the entire application at startup.

**How Lazy Loading Works:**

1. **Module Splitting**: Angular creates separate bundles for each lazy-loaded module
2. **On-demand Loading**: Modules are loaded when routes are accessed
3. **Code Splitting**: Webpack creates separate chunks for each module
4. **Dynamic Imports**: Uses ES2020 dynamic import() syntax

**Implementing Lazy Loading:**

```typescript
// app-routing.module.ts
const routes: Routes = [
  { path: '', component: HomeComponent },
  {
    path: 'users',
    loadChildren: () => import('./users/users.module').then((m) => m.UsersModule),
  },
  {
    path: 'products',
    loadChildren: () => import('./products/products.module').then((m) => m.ProductsModule),
  },
  {
    path: 'admin',
    loadChildren: () => import('./admin/admin.module').then((m) => m.AdminModule),
    canLoad: [AdminGuard], // Prevent loading if not authorized
  },
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule],
})
export class AppRoutingModule {}
```

**Lazy Loading with Route Guards:**

```typescript
// admin.guard.ts
@Injectable({
  providedIn: 'root',
})
export class AdminGuard implements CanLoad {
  constructor(
    private authService: AuthService,
    private router: Router,
  ) {}

  canLoad(route: Route, segments: UrlSegment[]): boolean | Observable<boolean> | Promise<boolean> {
    if (this.authService.isAdmin()) {
      return true;
    }

    this.router.navigate(['/unauthorized']);
    return false;
  }
}
```

**Preloading Strategies:**

1. **No Preloading** (Default):

```typescript
@NgModule({
  imports: [RouterModule.forRoot(routes, { preloadingStrategy: NoPreloading })],
})
export class AppRoutingModule {}
```

2. **Preload All Modules**:

```typescript
@NgModule({
  imports: [RouterModule.forRoot(routes, { preloadingStrategy: PreloadAllModules })],
})
export class AppRoutingModule {}
```

3. **Custom Preloading Strategy**:

```typescript
@Injectable()
export class CustomPreloadingStrategy implements PreloadingStrategy {
  preload(route: Route, load: Function): Observable<any> {
    // Preload based on custom logic
    if (route.data && route.data['preload']) {
      return load();
    }
    return of(null);
  }
}

// Usage
@NgModule({
  imports: [RouterModule.forRoot(routes, { preloadingStrategy: CustomPreloadingStrategy })]
})
```

**Route Configuration with Data:**

```typescript
const routes: Routes = [
  {
    path: 'dashboard',
    loadChildren: () => import('./dashboard/dashboard.module').then((m) => m.DashboardModule),
    data: { preload: true, delay: 1000 },
  },
  {
    path: 'settings',
    loadChildren: () => import('./settings/settings.module').then((m) => m.SettingsModule),
    data: { preload: false },
  },
];
```

**Benefits of Lazy Loading:**

1. **Improved Initial Load Time**:
   - Smaller initial bundle size
   - Faster application startup
   - Better user experience

2. **Better Performance**:
   - Reduced memory usage
   - Faster route transitions
   - Optimized resource utilization

3. **Code Organization**:
   - Clear module boundaries
   - Better separation of concerns
   - Easier maintenance

4. **Bandwidth Optimization**:
   - Users only download needed code
   - Reduced data usage
   - Better for mobile users

**Performance Monitoring:**

```typescript
// Performance monitoring
interface ModuleLoadTime {
  moduleName: string;
  loadTime: number;
  timestamp: Date;
}

@Injectable()
export class PerformanceService {
  private loadTimes: ModuleLoadTime[] = [];

  recordModuleLoad(moduleName: string, loadTime: number) {
    this.loadTimes.push({
      moduleName,
      loadTime,
      timestamp: new Date(),
    });
  }

  getLoadTimes(): ModuleLoadTime[] {
    return this.loadTimes;
  }
}
```

**Best Practices:**

- Use lazy loading for large feature modules
- Implement appropriate preloading strategies
- Monitor bundle sizes and load times
- Use route guards for conditional loading
- Consider user behavior patterns for preloading decisions

### Q18: What is the difference between forRoot() and forChild() in Angular modules?

**Answer:**
`forRoot()` and `forChild()` are static methods used in Angular modules to provide different configurations based on whether the module is being imported at the root level or as a child module.

**Purpose:**

- **forRoot()**: Used when a module is imported at the root level (typically in AppModule)
- **forChild()**: Used when a module is imported as a child module (feature modules)

**Common Examples:**

1. **RouterModule**:

```typescript
// app.module.ts (Root level)
import { RouterModule } from '@angular/router';

@NgModule({
  imports: [
    RouterModule.forRoot(routes, {
      enableTracing: false, // Disable in production
      useHash: false,
    }),
  ],
})
export class AppModule {}

// feature.module.ts (Child level)
import { RouterModule } from '@angular/router';

@NgModule({
  imports: [RouterModule.forChild(featureRoutes)],
})
export class FeatureModule {}
```

2. **HttpClientModule**:

```typescript
// app.module.ts
import { HttpClientModule } from '@angular/common/http';

@NgModule({
  imports: [
    HttpClientModule, // No forRoot() needed
  ],
})
export class AppModule {}

// feature.module.ts
import { HttpClientModule } from '@angular/common/http';

@NgModule({
  imports: [
    HttpClientModule, // Same import, but creates child injector
  ],
})
export class FeatureModule {}
```

3. **Custom Module with forRoot/forChild**:

```typescript
// config.module.ts
@NgModule({})
export class ConfigModule {
  static forRoot(config: Config): ModuleWithProviders<ConfigModule> {
    return {
      ngModule: ConfigModule,
      providers: [{ provide: CONFIG_TOKEN, useValue: config }, ConfigService],
    };
  }

  static forChild(): ModuleWithProviders<ConfigModule> {
    return {
      ngModule: ConfigModule,
      providers: [
        // Child-specific providers
        { provide: CONFIG_TOKEN, useValue: {} },
      ],
    };
  }
}

// app.module.ts
@NgModule({
  imports: [
    ConfigModule.forRoot({
      apiUrl: 'https://api.example.com',
      timeout: 5000,
    }),
  ],
})
export class AppModule {}

// feature.module.ts
@NgModule({
  imports: [ConfigModule.forChild()],
})
export class FeatureModule {}
```

**RouterModule Implementation Example:**

```typescript
// router_module.ts (simplified)
export class RouterModule {
  static forRoot(routes: Routes, config?: ExtraOptions): ModuleWithProviders<RouterModule> {
    return {
      ngModule: RouterModule,
      providers: [
        ROUTES,
        provideRoutes(routes),
        Router,
        { provide: ActivatedRoute, useFactory: rootRoute, deps: [Router] },
        RouterPreloader,
        NoPreloading,
        PreloadAllModules,
        { provide: ROUTER_CONFIGURATION, useValue: config ? config : {} },
        {
          provide: LocationStrategy,
          useFactory: provideLocationStrategy,
          deps: [PlatformLocation, ROUTER_CONFIGURATION],
        },
        ViewportScroller,
        { provide: RouterScroller, useExisting: ViewportScroller },
        {
          provide: RouterEvent,
          useValue: new NavigationStart(0, ''),
        },
      ],
    };
  }

  static forChild(routes: Routes): ModuleWithProviders<RouterModule> {
    return {
      ngModule: RouterModule,
      providers: [provideRoutes(routes)],
    };
  }
}
```

**Key Differences:**

| Aspect            | forRoot()                       | forChild()                     |
| ----------------- | ------------------------------- | ------------------------------ |
| **Usage**         | Root module only                | Feature modules                |
| **Providers**     | Creates root-level providers    | Creates child-level providers  |
| **Configuration** | Global configuration            | Feature-specific configuration |
| **Instance**      | Single instance per application | Multiple instances possible    |
| **Services**      | Singleton services              | May create new instances       |

**When to Use Each:**

1. **Use forRoot() when:**
   - Configuring global services
   - Setting up application-wide routing
   - Providing singleton services
   - Importing at the root level

2. **Use forChild() when:**
   - Importing feature modules
   - Needing feature-specific configuration
   - Creating isolated service instances
   - Adding feature-specific routes

**Best Practices:**

- Only call forRoot() once in your application
- Use forChild() for all feature module imports
- Understand the provider scope implications
- Document when to use each method in custom modules

### Q19: How do you create shared modules in Angular?

**Answer:**
Shared modules contain components, directives, pipes, and other functionality that are used across multiple feature modules in an Angular application.

**Purpose of Shared Modules:**

- Reusable components and utilities
- Common pipes and directives
- Shared services (if needed)
- Third-party library integration
- Consistent UI patterns

**Creating a Shared Module:**

```typescript
// shared.module.ts
import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';

// Angular modules
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { HttpClientModule } from '@angular/common/http';

// Third-party modules
import { NgxChartsModule } from '@swimlane/ngx-charts';
import { NgxPaginationModule } from 'ngx-pagination';

// Components
import { LoadingComponent } from './components/loading/loading.component';
import { ErrorComponent } from './components/error/error.component';
import { ConfirmDialogComponent } from './components/confirm-dialog/confirm-dialog.component';
import { SearchBoxComponent } from './components/search-box/search-box.component';

// Directives
import { HighlightDirective } from './directives/highlight.directive';
import { TooltipDirective } from './directives/tooltip.directive';

// Pipes
import { CurrencyFormatPipe } from './pipes/currency-format.pipe';
import { DateFormatPipe } from './pipes/date-format.pipe';
import { TruncatePipe } from './pipes/truncate.pipe';

// Services
import { NotificationService } from './services/notification.service';
import { LoadingService } from './services/loading.service';

@NgModule({
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,
    HttpClientModule,
    NgxChartsModule,
    NgxPaginationModule,
  ],
  declarations: [
    LoadingComponent,
    ErrorComponent,
    ConfirmDialogComponent,
    SearchBoxComponent,
    HighlightDirective,
    TooltipDirective,
    CurrencyFormatPipe,
    DateFormatPipe,
    TruncatePipe,
  ],
  exports: [
    // Re-export Angular modules
    CommonModule,
    FormsModule,
    ReactiveFormsModule,
    HttpClientModule,

    // Re-export third-party modules
    NgxChartsModule,
    NgxPaginationModule,

    // Re-export components
    LoadingComponent,
    ErrorComponent,
    ConfirmDialogComponent,
    SearchBoxComponent,

    // Re-export directives
    HighlightDirective,
    TooltipDirective,

    // Re-export pipes
    CurrencyFormatPipe,
    DateFormatPipe,
    TruncatePipe,
  ],
  providers: [NotificationService, LoadingService],
})
export class SharedModule {}
```

**Shared Module Structure:**

```
shared/
├── components/
│   ├── loading/
│   ├── error/
│   ├── confirm-dialog/
│   └── search-box/
├── directives/
│   ├── highlight.directive.ts
│   └── tooltip.directive.ts
├── pipes/
│   ├── currency-format.pipe.ts
│   ├── date-format.pipe.ts
│   └── truncate.pipe.ts
├── services/
│   ├── notification.service.ts
│   └── loading.service.ts
├── models/
│   ├── notification.model.ts
│   └── loading.model.ts
├── shared.module.ts
└── index.ts
```

**Using Shared Module in Feature Modules:**

```typescript
// user.module.ts
import { NgModule } from '@angular/core';
import { SharedModule } from '../shared/shared.module';
import { UserRoutingModule } from './user-routing.module';

// Feature-specific components
import { UserListComponent } from './components/user-list/user-list.component';
import { UserFormComponent } from './components/user-form/user-form.component';

@NgModule({
  declarations: [UserListComponent, UserFormComponent],
  imports: [
    SharedModule, // Import shared functionality
    UserRoutingModule,
  ],
})
export class UserModule {}
```

**Shared Module Best Practices:**

1. **Selective Exports**:

```typescript
// Only export what's needed
exports: [
  CommonModule, // Essential for ngIf, ngFor
  FormsModule, // If forms are commonly used
  LoadingComponent, // Reusable component
  CurrencyFormatPipe, // Common pipe
];
```

2. **Avoid Service Duplication**:

```typescript
// Option 1: Provide in root (recommended for most services)
@Injectable({
  providedIn: 'root'
})
export class NotificationService {}

// Option 2: Provide in shared module (if feature-specific)
@NgModule({
  providers: [NotificationService] // Creates instance per importing module
})
export class SharedModule {}

// Option 3: Use forRoot pattern for configuration
static forRoot(config: NotificationConfig): ModuleWithProviders<SharedModule> {
  return {
    ngModule: SharedModule,
    providers: [
      NotificationService,
      { provide: NOTIFICATION_CONFIG, useValue: config }
    ]
  };
}
```

3. **Third-party Integration**:

```typescript
// shared.module.ts
import { ChartsModule } from 'ng2-charts';

@NgModule({
  imports: [ChartsModule],
  exports: [ChartsModule],
})
export class SharedModule {}
```

4. **Barrel Export**:

```typescript
// shared/index.ts
export * from './components/loading/loading.component';
export * from './components/error/error.component';
export * from './directives/highlight.directive';
export * from './pipes/currency-format.pipe';
export * from './services/notification.service';
export * from './shared.module';
```

**Common Shared Module Patterns:**

1. **UI Components**:

```typescript
// ui.module.ts
@NgModule({
  declarations: [ButtonComponent, InputComponent, ModalComponent, CardComponent],
  exports: [ButtonComponent, InputComponent, ModalComponent, CardComponent],
})
export class UiModule {}
```

2. **Pipes Module**:

```typescript
// pipes.module.ts
@NgModule({
  declarations: [CurrencyFormatPipe, DateFormatPipe, TruncatePipe, FilterPipe],
  exports: [CurrencyFormatPipe, DateFormatPipe, TruncatePipe, FilterPipe],
})
export class PipesModule {}
```

3. **Directives Module**:

```typescript
// directives.module.ts
@NgModule({
  declarations: [HighlightDirective, TooltipDirective, ClickOutsideDirective],
  exports: [HighlightDirective, TooltipDirective, ClickOutsideDirective],
})
export class DirectivesModule {}
```

**Troubleshooting Common Issues:**

1. **Circular Dependencies**:

```typescript
// Avoid importing feature modules in shared module
// shared.module.ts should not import user.module.ts
```

2. **Service Duplication**:

```typescript
// Use providedIn: 'root' for singleton services
// Or use forRoot pattern for configurable services
```

3. **Import Order**:

```typescript
// Import shared module before feature modules
@NgModule({
  imports: [
    SharedModule, // Import first
    UserModule,   // Then feature modules
    AdminModule
  ]
})
```

**Best Practices:**

- Keep shared modules focused and cohesive
- Use barrel exports for easier importing
- Document shared module usage
- Avoid feature-specific logic in shared modules
- Regularly review and refactor shared components

---

## 5. Data Binding & Templates

### Q20: Explain the different types of data binding in Angular

**Answer:**
Angular provides four types of data binding that facilitate communication between the component class and the template:

**1. Interpolation ({{ }}) - One-way from Component to Template:**

```typescript
// Component
export class UserComponent {
  userName = 'John Doe';
  userAge = 30;
  currentDate = new Date();
}

// Template
<h1>Welcome, {{ userName }}!</h1>
<p>Age: {{ userAge }}</p>
<p>Today's date: {{ currentDate | date:'fullDate' }}</p>
<p>Calculation: {{ userAge * 2 }}</p>
```

**2. Property Binding ([property]) - One-way from Component to Template:**

```typescript
// Component
export class ButtonComponent {
  isDisabled = false;
  buttonText = 'Click me';
  imageUrl = 'assets/logo.png';
  cssClass = 'btn-primary';
}

// Template
<button [disabled]="isDisabled" [innerText]="buttonText">Default</button>
<img [src]="imageUrl" [alt]="buttonText">
<div [class]="cssClass">Styled div</div>
<app-user-card [user]="currentUser" [showDetails]="true"></app-user-card>
```

**3. Event Binding ((event)) - One-way from Template to Component:**

```typescript
// Component
export class FormComponent {
  handleSubmit(event: Event) {
    console.log('Form submitted', event);
  }

  handleClick() {
    console.log('Button clicked');
  }

  handleInput(event: Event) {
    const target = event.target as HTMLInputElement;
    console.log('Input value:', target.value);
  }
}

// Template
<form (submit)="handleSubmit($event)">
  <input (input)="handleInput($event)" placeholder="Type here">
  <button (click)="handleClick()">Submit</button>
</form>
```

**4. Two-way Binding ([(ngModel)]) - Bidirectional:**

```typescript
// Component (requires FormsModule)
export class UserFormComponent {
  userName = 'Initial Value';
  userSettings = {
    notifications: true,
    darkMode: false
  };
}

// Template
<input [(ngModel)]="userName" placeholder="Enter name">
<p>Current value: {{ userName }}</p>

<label>
  <input type="checkbox" [(ngModel)]="userSettings.notifications">
  Enable notifications
</label>

<label>
  <input type="checkbox" [(ngModel)]="userSettings.darkMode">
  Dark mode
</label>
```

**Advanced Binding Examples:**

1. **Safe Navigation Operator (?.):**

```typescript
// Component
export class UserProfileComponent {
  user: User | null = null; // Could be null
}

// Template
<p>User name: {{ user?.name }}</p>
<p>User email: {{ user?.contact?.email }}</p>
<img [src]="user?.avatarUrl" alt="User avatar">
```

2. **Template Reference Variables (#):**

```typescript
// Template
<input #nameInput type="text" placeholder="Enter name">
<button (click)="submitForm(nameInput.value)">Submit</button>

<form #userForm="ngForm">
  <input name="email" ngModel>
  <button [disabled]="!userForm.valid">Submit</button>
</form>
```

3. **Attribute Binding:**

```typescript
// Template
<div [attr.data-id]="userId">User ID: {{ userId }}</div>
<button [attr.aria-label]="buttonLabel">Action</button>
<table [attr.colspan]="columns.length">
```

4. **Style Binding:**

```typescript
// Component
export class StyleComponent {
  isHighlighted = true;
  fontSize = 16;
  backgroundColor = 'lightblue';
}

// Template
<div [style.font-size.px]="fontSize">Text with dynamic font size</div>
<div [style.background-color]="backgroundColor">Colored background</div>
<div [ngStyle]="{ 'font-weight': isHighlighted ? 'bold' : 'normal' }">Styled text</div>
```

5. **Class Binding:**

```typescript
// Component
export class ClassComponent {
  isActive = true;
  hasError = false;
  cssClasses = {
    'active': this.isActive,
    'error': this.hasError,
    'disabled': !this.isActive
  };
}

// Template
<div [class.active]="isActive">Active state</div>
<div [class.error]="hasError">Error state</div>
<div [ngClass]="cssClasses">Multiple classes</div>
```

**Binding Performance Considerations:**

1. **Avoid Complex Expressions:**

```typescript
// ❌ Avoid complex expressions in templates
<p>{{ expensiveCalculation() }}</p>

// ✅ Use computed properties or methods
<p>{{ calculatedValue }}</p>

// Component
get calculatedValue(): string {
  return this.expensiveCalculation();
}
```

2. **Use OnPush Change Detection:**

```typescript
@Component({
  selector: 'app-user-list',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div *ngFor="let user of users">
      {{ user.name }}
    </div>
  `,
})
export class UserListComponent {
  @Input() users: User[];
}
```

3. **Memoization for Expensive Operations:**

```typescript
export class ExpensiveCalculationComponent {
  private _cache = new Map<string, any>();

  getCachedValue(key: string): any {
    if (!this._cache.has(key)) {
      this._cache.set(key, this.expensiveOperation(key));
    }
    return this._cache.get(key);
  }
}
```

**Best Practices:**

- Use property binding for dynamic attributes
- Use event binding for user interactions
- Use two-way binding sparingly (can impact performance)
- Use safe navigation operator for nullable properties
- Keep template expressions simple and fast
- Use trackBy with ngFor for better performance
- Prefer OnPush change detection when possible

### Q21: How do you handle forms in Angular (Template-driven vs Reactive)?

**Answer:**
Angular provides two approaches for handling forms: Template-driven and Reactive forms, each with different use cases and patterns.

**Template-Driven Forms:**

**Setup:**

```typescript
// app.module.ts
import { FormsModule } from '@angular/forms';

@NgModule({
  imports: [FormsModule],
  // ...
})
export class AppModule {}
```

**Basic Template-Driven Form:**

```typescript
// Component
export class UserFormComponent {
  user = {
    name: '',
    email: '',
    password: '',
  };

  onSubmit(form: NgForm) {
    if (form.valid) {
      console.log('Form submitted:', this.user);
    }
  }
}
```

```html
<!-- Template -->
<form #userForm="ngForm" (ngSubmit)="onSubmit(userForm)">
  <div>
    <label for="name">Name:</label>
    <input
      id="name"
      name="name"
      [(ngModel)]="user.name"
      required
      minlength="2"
      #nameInput="ngModel" />
    <div *ngIf="nameInput.invalid && nameInput.touched" class="error">
      Name is required and must be at least 2 characters
    </div>
  </div>

  <div>
    <label for="email">Email:</label>
    <input
      id="email"
      name="email"
      type="email"
      [(ngModel)]="user.email"
      required
      email
      #emailInput="ngModel" />
    <div *ngIf="emailInput.invalid && emailInput.touched" class="error">
      Please enter a valid email
    </div>
  </div>

  <div>
    <label for="password">Password:</label>
    <input
      id="password"
      name="password"
      type="password"
      [(ngModel)]="user.password"
      required
      minlength="6"
      #passwordInput="ngModel" />
    <div *ngIf="passwordInput.invalid && passwordInput.touched" class="error">
      Password must be at least 6 characters
    </div>
  </div>

  <button type="submit" [disabled]="userForm.invalid">Submit</button>
</form>
```

**Reactive Forms:**

**Setup:**

```typescript
// app.module.ts
import { ReactiveFormsModule } from '@angular/forms';

@NgModule({
  imports: [ReactiveFormsModule],
  // ...
})
export class AppModule {}
```

**Basic Reactive Form:**

```typescript
// Component
import { FormBuilder, FormGroup, Validators } from '@angular/forms';

export class UserFormComponent implements OnInit {
  userForm: FormGroup;

  constructor(private fb: FormBuilder) {}

  ngOnInit() {
    this.userForm = this.fb.group(
      {
        name: ['', [Validators.required, Validators.minLength(2)]],
        email: ['', [Validators.required, Validators.email]],
        password: ['', [Validators.required, Validators.minLength(6)]],
        confirmPassword: ['', Validators.required],
      },
      {
        validators: this.passwordMatchValidator,
      },
    );
  }

  passwordMatchValidator(form: FormGroup) {
    const password = form.get('password')?.value;
    const confirmPassword = form.get('confirmPassword')?.value;

    return password === confirmPassword ? null : { passwordMismatch: true };
  }

  onSubmit() {
    if (this.userForm.valid) {
      console.log('Form submitted:', this.userForm.value);
    }
  }

  get name() {
    return this.userForm.get('name');
  }

  get email() {
    return this.userForm.get('email');
  }

  get password() {
    return this.userForm.get('password');
  }
}
```

```html
<!-- Template -->
<form [formGroup]="userForm" (ngSubmit)="onSubmit()">
  <div>
    <label for="name">Name:</label>
    <input id="name" formControlName="name" />
    <div *ngIf="name?.invalid && name?.touched" class="error">
      Name is required and must be at least 2 characters
    </div>
  </div>

  <div>
    <label for="email">Email:</label>
    <input id="email" formControlName="email" type="email" />
    <div *ngIf="email?.invalid && email?.touched" class="error">Please enter a valid email</div>
  </div>

  <div>
    <label for="password">Password:</label>
    <input id="password" formControlName="password" type="password" />
    <div *ngIf="password?.invalid && password?.touched" class="error">
      Password must be at least 6 characters
    </div>
  </div>

  <div>
    <label for="confirmPassword">Confirm Password:</label>
    <input id="confirmPassword" formControlName="confirmPassword" type="password" />
    <div *ngIf="userForm.hasError('passwordMismatch')" class="error">Passwords do not match</div>
  </div>

  <button type="submit" [disabled]="userForm.invalid">Submit</button>
</form>
```

**Advanced Reactive Forms:**

1. **Dynamic Form Controls:**

```typescript
export class DynamicFormComponent implements OnInit {
  form: FormGroup;

  constructor(private fb: FormBuilder) {}

  ngOnInit() {
    this.form = this.fb.group({
      items: this.fb.array([this.createItem()]),
    });
  }

  createItem(): FormGroup {
    return this.fb.group({
      name: ['', Validators.required],
      quantity: [1, [Validators.required, Validators.min(1)]],
      price: [0, Validators.required],
    });
  }

  get items(): FormArray {
    return this.form.get('items') as FormArray;
  }

  addItem() {
    this.items.push(this.createItem());
  }

  removeItem(index: number) {
    this.items.removeAt(index);
  }

  calculateTotal(): number {
    return this.items.controls.reduce((sum, control) => {
      const item = control.value;
      return sum + item.quantity * item.price;
    }, 0);
  }
}
```

```html
<!-- Template -->
<form [formGroup]="form">
  <div formArrayName="items">
    <div *ngFor="let item of items.controls; let i = index" [formGroupName]="i">
      <input formControlName="name" placeholder="Item name" />
      <input formControlName="quantity" type="number" placeholder="Quantity" />
      <input formControlName="price" type="number" placeholder="Price" />
      <button (click)="removeItem(i)">Remove</button>
    </div>
  </div>

  <button (click)="addItem()">Add Item</button>
  <p>Total: ${{ calculateTotal() }}</p>
</form>
```

2. **Custom Validators:**

```typescript
// Custom validator function
export function forbiddenNameValidator(forbiddenName: string): ValidatorFn {
  return (control: AbstractControl): ValidationErrors | null => {
    const forbidden = new RegExp(forbiddenName, 'i').test(control.value);
    return forbidden ? { forbiddenName: { value: control.value } } : null;
  };
}

// Async validator
export function uniqueEmailValidator(userService: UserService): AsyncValidatorFn {
  return (control: AbstractControl): Observable<ValidationErrors | null> => {
    return userService.checkEmailExists(control.value).pipe(
      map((exists) => (exists ? { emailExists: true } : null)),
      catchError(() => of(null)),
    );
  };
}

// Usage in component
this.userForm = this.fb.group({
  name: ['', [Validators.required, forbiddenNameValidator('admin')]],
  email: ['', [Validators.required, Validators.email], [uniqueEmailValidator(this.userService)]],
});
```

3. **Form State Management:**

```typescript
export class FormStateComponent implements OnInit {
  userForm: FormGroup;

  constructor(private fb: FormBuilder) {}

  ngOnInit() {
    this.userForm = this.fb.group({
      name: ['', Validators.required],
      email: ['', [Validators.required, Validators.email]],
      age: [18, [Validators.min(18), Validators.max(65)]],
    });

    // Watch for form changes
    this.userForm.valueChanges.subscribe((value) => {
      console.log('Form changed:', value);
    });

    // Watch for specific control changes
    this.userForm.get('name')?.valueChanges.subscribe((value) => {
      console.log('Name changed:', value);
    });

    // Watch for form status changes
    this.userForm.statusChanges.subscribe((status) => {
      console.log('Form status:', status);
    });
  }

  // Conditional validation
  toggleAgeValidation(required: boolean) {
    const ageControl = this.userForm.get('age');
    if (required) {
      ageControl?.setValidators([Validators.required, Validators.min(18)]);
    } else {
      ageControl?.clearValidators();
    }
    ageControl?.updateValueAndValidity();
  }
}
```

**Template-Driven vs Reactive Forms Comparison:**

| Aspect             | Template-Driven          | Reactive               |
| ------------------ | ------------------------ | ---------------------- |
| **Setup**          | Simple, less code        | More setup, explicit   |
| **Testing**        | Harder to test           | Easier to test         |
| **Validation**     | Template-based           | Code-based             |
| **Dynamic Forms**  | Limited                  | Full support           |
| **Performance**    | Slower for complex forms | Better performance     |
| **Flexibility**    | Less flexible            | Highly flexible        |
| **Learning Curve** | Easier for beginners     | Steeper learning curve |

**When to Use Each:**

**Template-Driven Forms:**

- Simple forms with basic validation
- Rapid prototyping
- Forms with minimal dynamic behavior
- When you prefer template-centric approach

**Reactive Forms:**

- Complex forms with dynamic behavior
- Forms requiring custom validation
- When you need better testability
- Forms with conditional logic
- When you prefer code-centric approach

**Best Practices:**

1. **Choose the right approach** based on form complexity
2. **Use FormBuilder** to reduce boilerplate code
3. **Create custom validators** for complex validation rules
4. **Handle form state** properly (pristine, touched, dirty)
5. **Use reactive patterns** for dynamic form behavior
6. **Implement proper error handling** and user feedback
7. **Consider accessibility** in form design
8. **Optimize performance** for large forms with OnPush change detection

### Q22: What are Angular directives and how do you create custom directives?

**Answer:**
Directives are Angular features that allow you to extend HTML with custom behavior and manipulate the DOM. There are three types of directives in Angular: Components, Attribute Directives, and Structural Directives.

**Types of Directives:**

1. **Components**: Directives with a template (extends directives)
2. **Attribute Directives**: Change the appearance or behavior of elements
3. **Structural Directives**: Change the DOM layout by adding/removing elements

**Creating Attribute Directives:**

**Basic Attribute Directive:**

```typescript
import { Directive, ElementRef, Renderer2, OnInit, HostListener, HostBinding } from '@angular/core';

@Directive({
  selector: '[appHighlight]',
})
export class HighlightDirective implements OnInit {
  constructor(
    private el: ElementRef,
    private renderer: Renderer2,
  ) {}

  ngOnInit() {
    this.renderer.setStyle(this.el.nativeElement, 'backgroundColor', 'yellow');
  }

  @HostListener('mouseenter') onMouseEnter() {
    this.renderer.setStyle(this.el.nativeElement, 'backgroundColor', 'orange');
  }

  @HostListener('mouseleave') onMouseLeave() {
    this.renderer.setStyle(this.el.nativeElement, 'backgroundColor', 'yellow');
  }
}
```

```html
<!-- Usage -->
<p appHighlight>This text will be highlighted on hover</p>
```

**Attribute Directive with Input:**

```typescript
import { Directive, ElementRef, Renderer2, OnInit, Input, HostListener } from '@angular/core';

@Directive({
  selector: '[appColor]',
})
export class ColorDirective implements OnInit {
  @Input() appColor = 'yellow';
  @Input() appColorHover = 'orange';

  constructor(
    private el: ElementRef,
    private renderer: Renderer2,
  ) {}

  ngOnInit() {
    this.setColor(this.appColor);
  }

  @HostListener('mouseenter') onMouseEnter() {
    this.setColor(this.appColorHover);
  }

  @HostListener('mouseleave') onMouseLeave() {
    this.setColor(this.appColor);
  }

  private setColor(color: string) {
    this.renderer.setStyle(this.el.nativeElement, 'backgroundColor', color);
  }
}
```

```html
<!-- Usage -->
<p appColor="lightblue" appColorHover="blue">Custom colors</p>
```

**Advanced Attribute Directive with Multiple Inputs:**

```typescript
import {
  Directive,
  ElementRef,
  Renderer2,
  OnInit,
  Input,
  OnChanges,
  SimpleChanges,
} from '@angular/core';

@Directive({
  selector: '[appStyle]',
})
export class StyleDirective implements OnInit, OnChanges {
  @Input() appStyle: { [key: string]: string } = {};

  constructor(
    private el: ElementRef,
    private renderer: Renderer2,
  ) {}

  ngOnInit() {
    this.applyStyles();
  }

  ngOnChanges(changes: SimpleChanges) {
    if (changes['appStyle']) {
      this.applyStyles();
    }
  }

  private applyStyles() {
    Object.keys(this.appStyle).forEach((property) => {
      this.renderer.setStyle(this.el.nativeElement, property, this.appStyle[property]);
    });
  }
}
```

```html
<!-- Usage -->
<div [appStyle]="{ 'background-color': 'red', 'color': 'white', 'padding': '10px' }">
  Styled div
</div>
```

**Creating Structural Directives:**

**Custom Structural Directive:**

```typescript
import { Directive, Input, TemplateRef, ViewContainerRef } from '@angular/core';

@Directive({
  selector: '[appUnless]',
})
export class UnlessDirective {
  private hasView = false;

  constructor(
    private templateRef: TemplateRef<any>,
    private viewContainer: ViewContainerRef,
  ) {}

  @Input() set appUnless(condition: boolean) {
    if (!condition && !this.hasView) {
      this.viewContainer.createEmbeddedView(this.templateRef);
      this.hasView = true;
    } else if (condition && this.hasView) {
      this.viewContainer.clear();
      this.hasView = false;
    }
  }
}
```

```html
<!-- Usage -->
<div *appUnless="userLoggedIn">Please log in to continue</div>
```

**Advanced Structural Directive with Context:**

```typescript
import { Directive, Input, TemplateRef, ViewContainerRef, EmbeddedViewRef } from '@angular/core';

interface ForContext<T> {
  $implicit: T;
  index: number;
  count: number;
  first: boolean;
  last: boolean;
  even: boolean;
  odd: boolean;
}

@Directive({
  selector: '[appFor]',
})
export class ForDirective<T> {
  private views: EmbeddedViewRef<ForContext<T>>[] = [];

  constructor(
    private templateRef: TemplateRef<ForContext<T>>,
    private viewContainer: ViewContainerRef,
  ) {}

  @Input() set appForOf(items: T[]) {
    // Clear existing views
    this.views.forEach((view) => view.destroy());
    this.views = [];

    // Create new views
    items.forEach((item, index) => {
      const context: ForContext<T> = {
        $implicit: item,
        index,
        count: items.length,
        first: index === 0,
        last: index === items.length - 1,
        even: index % 2 === 0,
        odd: index % 2 !== 0,
      };

      const view = this.viewContainer.createEmbeddedView(this.templateRef, context);
      this.views.push(view);
    });
  }
}
```

```html
<!-- Usage -->
<div *appFor="items; let item; let i = index; let isFirst = first">
  {{ i + 1 }}. {{ item }}
  <span *ngIf="isFirst">(First item)</span>
</div>
```

**Directive with HostBinding and HostListener:**

```typescript
import { Directive, HostBinding, HostListener, Input } from '@angular/core';

@Directive({
  selector: '[appTooltip]',
})
export class TooltipDirective {
  @Input() appTooltip = '';
  @Input() tooltipPosition: 'top' | 'bottom' | 'left' | 'right' = 'top';

  @HostBinding('class.has-tooltip') hasTooltip = true;
  @HostBinding('attr.title') get tooltipText() {
    return this.appTooltip;
  }

  @HostListener('mouseenter') onMouseEnter() {
    console.log('Mouse entered tooltip area');
  }

  @HostListener('mouseleave') onMouseLeave() {
    console.log('Mouse left tooltip area');
  }
}
```

```html
<!-- Usage -->
<button appTooltip="This is a tooltip" tooltipPosition="bottom">Hover me</button>
```

**Directive with Dependency Injection:**

```typescript
import { Directive, ElementRef, Renderer2, OnInit, Inject } from '@angular/core';
import { DOCUMENT } from '@angular/common';

@Directive({
  selector: '[appExternalLink]',
})
export class ExternalLinkDirective implements OnInit {
  constructor(
    private el: ElementRef,
    private renderer: Renderer2,
    @Inject(DOCUMENT) private document: Document,
  ) {}

  ngOnInit() {
    const link = this.el.nativeElement;

    // Add external link attributes
    this.renderer.setAttribute(link, 'target', '_blank');
    this.renderer.setAttribute(link, 'rel', 'noopener noreferrer');

    // Add click tracking
    this.renderer.listen(link, 'click', (event) => {
      console.log('External link clicked:', link.href);
      // Track analytics here
    });
  }
}
```

```html
<!-- Usage -->
<a appExternalLink href="https://example.com">External Link</a>
```

**Directive Best Practices:**

1. **Use descriptive selectors:**

```typescript
// Good
@Directive({ selector: '[appHighlight]' })

// Avoid
@Directive({ selector: '[highlight]' })
```

2. **Handle cleanup properly:**

```typescript
@Directive({
  selector: '[appScrollTracker]',
})
export class ScrollTrackerDirective implements OnInit, OnDestroy {
  private scrollSubscription?: Subscription;

  ngOnInit() {
    this.scrollSubscription = fromEvent(window, 'scroll').subscribe(() => {
      // Track scroll position
    });
  }

  ngOnDestroy() {
    this.scrollSubscription?.unsubscribe();
  }
}
```

3. **Use HostBinding for host element styling:**

```typescript
@Directive({
  selector: '[appActive]',
})
export class ActiveDirective {
  @Input() isActive = false;

  @HostBinding('class.active') get activeClass() {
    return this.isActive;
  }
}
```

4. **Create reusable directives:**

```typescript
@Directive({
  selector: '[appClickOutside]',
})
export class ClickOutsideDirective {
  @Output() clickOutside = new EventEmitter<void>();

  @HostListener('document:click', ['$event'])
  onClick(event: Event) {
    if (!this.el.nativeElement.contains(event.target)) {
      this.clickOutside.emit();
    }
  }
}
```

**Common Use Cases for Custom Directives:**

1. **UI Enhancements**: Tooltips, highlights, animations
2. **Accessibility**: ARIA attributes, keyboard navigation
3. **Performance**: Lazy loading, virtualization
4. **Validation**: Custom form validation
5. **Behavior**: Drag and drop, click outside, scroll tracking
6. **Styling**: Dynamic classes, conditional styling

**Directive vs Component Decision:**

**Use Directives when:**

- You need to extend existing elements
- You're adding behavior to elements
- You need to manipulate the DOM
- You want to create reusable behaviors

**Use Components when:**

- You need to create new UI elements
- You need complex templates
- You need to encapsulate functionality
- You need to create reusable UI blocks

**Best Practices:**

- Keep directives focused on a single responsibility
- Use meaningful names with app prefix
- Handle cleanup in ngOnDestroy
- Use HostBinding and HostListener for host element interaction
- Consider performance implications for frequently triggered directives
- Document directive usage and behavior clearly

---

## 6. Routing & Navigation

### Q23: Explain Angular routing and how to implement complex routing scenarios

**Answer:**
Angular routing provides a way to navigate between different views/components in a single-page application (SPA) without full page reloads.

**Basic Routing Setup:**

```typescript
// app-routing.module.ts
import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { HomeComponent } from './components/home/home.component';
import { AboutComponent } from './components/about/about.component';
import { NotFoundComponent } from './components/not-found/not-found.component';

const routes: Routes = [
  { path: '', component: HomeComponent },
  { path: 'about', component: AboutComponent },
  { path: '404', component: NotFoundComponent },
  { path: '**', redirectTo: '/404' },
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule],
})
export class AppRoutingModule {}
```

**Route Parameters:**

```typescript
// user-routing.module.ts
const routes: Routes = [
  { path: 'users', component: UserListComponent },
  { path: 'users/:id', component: UserDetailComponent },
  { path: 'users/:id/edit', component: UserEditComponent },
];

// Component using route parameters
export class UserDetailComponent implements OnInit {
  userId: string;

  constructor(private route: ActivatedRoute) {}

  ngOnInit() {
    // Get route parameter
    this.userId = this.route.snapshot.paramMap.get('id');

    // Reactive approach for parameter changes
    this.route.paramMap.subscribe((params) => {
      this.userId = params.get('id');
      this.loadUser();
    });
  }
}
```

**Query Parameters:**

```typescript
// Setting query parameters
this.router.navigate(['/users'], {
  queryParams: { filter: 'active', page: 2 },
  queryParamsHandling: 'merge', // Keep existing query params
});

// Reading query parameters
this.route.queryParamMap.subscribe((params) => {
  const filter = params.get('filter');
  const page = params.get('page');
});
```

**Route Guards:**

1. **CanActivate Guard:**

```typescript
@Injectable({
  providedIn: 'root',
})
export class AuthGuard implements CanActivate {
  constructor(
    private authService: AuthService,
    private router: Router,
  ) {}

  canActivate(
    route: ActivatedRouteSnapshot,
    state: RouterStateSnapshot,
  ): Observable<boolean | UrlTree> | Promise<boolean | UrlTree> | boolean | UrlTree {
    if (this.authService.isLoggedIn()) {
      return true;
    } else {
      this.router.navigate(['/login'], {
        queryParams: { returnUrl: state.url },
      });
      return false;
    }
  }
}
```

2. **CanDeactivate Guard:**

```typescript
@Injectable({
  providedIn: 'root',
})
export class UnsavedChangesGuard implements CanDeactivate<UserEditComponent> {
  canDeactivate(
    component: UserEditComponent,
    currentRoute: ActivatedRouteSnapshot,
    currentState: RouterStateSnapshot,
    nextState?: RouterStateSnapshot,
  ): Observable<boolean | UrlTree> | Promise<boolean | UrlTree> | boolean | UrlTree {
    if (component.hasUnsavedChanges()) {
      return confirm('You have unsaved changes. Are you sure you want to leave?');
    }
    return true;
  }
}
```

3. **Resolve Guard:**

```typescript
@Injectable({
  providedIn: 'root',
})
export class UserResolver implements Resolve<User> {
  constructor(private userService: UserService) {}

  resolve(route: ActivatedRouteSnapshot, state: RouterStateSnapshot): Observable<User> {
    const userId = route.paramMap.get('id');
    return this.userService.getUser(userId).pipe(
      catchError((error) => {
        this.router.navigate(['/404']);
        return of(null);
      }),
    );
  }
}
```

**Nested Routes and Child Routes:**

```typescript
// app-routing.module.ts
const routes: Routes = [
  {
    path: 'admin',
    component: AdminLayoutComponent,
    children: [
      { path: '', redirectTo: 'dashboard', pathMatch: 'full' },
      { path: 'dashboard', component: AdminDashboardComponent },
      {
        path: 'users',
        component: AdminUsersComponent,
        children: [{ path: ':id', component: AdminUserDetailComponent }],
      },
    ],
  },
];
```

**Named Outlets:**

```typescript
// Template with named outlets
<router-outlet></router-outlet>
<router-outlet name="sidebar"></router-outlet>

// Routes with named outlets
const routes: Routes = [
  {
    path: 'dashboard',
    component: DashboardComponent,
    outlet: 'sidebar'
  }
];

// Navigation to named outlet
this.router.navigate([
  { outlets: { primary: 'dashboard', sidebar: 'notifications' } }
]);
```

**Lazy Loading with Routing:**

```typescript
// app-routing.module.ts
const routes: Routes = [
  {
    path: 'users',
    loadChildren: () => import('./users/users.module').then((m) => m.UsersModule),
  },
  {
    path: 'admin',
    loadChildren: () => import('./admin/admin.module').then((m) => m.AdminModule),
    canLoad: [AdminGuard],
  },
];
```

**Route Animation:**

```typescript
// app-routing.module.ts
const routes: Routes = [
  {
    path: 'home',
    component: HomeComponent,
    data: { animation: 'home' },
  },
  {
    path: 'about',
    component: AboutComponent,
    data: { animation: 'about' },
  },
];

// Component with animations
@Component({
  selector: 'app-root',
  template: `
    <div [@routeAnimations]="prepareRoute(outlet)">
      <router-outlet #outlet="outlet"></router-outlet>
    </div>
  `,
  animations: [
    trigger('routeAnimations', [
      transition('* <=> *', [
        style({ opacity: 0 }),
        animate('500ms ease-in-out', style({ opacity: 1 })),
      ]),
    ]),
  ],
})
export class AppComponent {
  prepareRoute(outlet: RouterOutlet) {
    return outlet?.activatedRouteData?.['animation'];
  }
}
```

**Advanced Routing Patterns:**

1. **Conditional Routing:**

```typescript
// Dynamic route based on user role
const routes: Routes = [
  {
    path: 'dashboard',
    canActivate: [RoleGuard],
    data: { roles: ['admin', 'user'] },
    children: [
      {
        path: '',
        component: AdminDashboardComponent,
        data: { role: 'admin' },
      },
      {
        path: '',
        component: UserDashboardComponent,
        data: { role: 'user' },
      },
    ],
  },
];
```

2. **Route Preloading:**

```typescript
// Custom preloading strategy
@Injectable()
export class CustomPreloadingStrategy implements PreloadingStrategy {
  preload(route: Route, load: Function): Observable<any> {
    if (route.data && route.data['preload']) {
      return load();
    }
    return of(null);
  }
}

// Usage
@NgModule({
  imports: [RouterModule.forRoot(routes, {
    preloadingStrategy: CustomPreloadingStrategy
  })]
})
```

3. **Route Guards with Multiple Guards:**

```typescript
{
  path: 'protected',
  component: ProtectedComponent,
  canActivate: [AuthGuard, RoleGuard],
  canDeactivate: [UnsavedChangesGuard],
  resolve: { data: DataResolver }
}
```

**Router Events:**

```typescript
export class AppComponent implements OnInit {
  constructor(private router: Router) {}

  ngOnInit() {
    this.router.events.subscribe((event) => {
      if (event instanceof NavigationStart) {
        console.log('Navigation started:', event.url);
      }
      if (event instanceof NavigationEnd) {
        console.log('Navigation ended:', event.urlAfterRedirects);
      }
      if (event instanceof NavigationError) {
        console.error('Navigation error:', event.error);
      }
    });
  }
}
```

**Best Practices:**

- Use route guards for authentication and authorization
- Implement resolve guards for data preloading
- Use lazy loading for better performance
- Handle navigation errors gracefully
- Use named outlets for complex layouts
- Implement proper error handling for missing routes
- Consider route animations for better UX

### Q24: How do you handle HTTP requests and responses in Angular?

**Answer:**
Angular provides the HttpClient module for making HTTP requests to RESTful APIs and handling responses.

**HttpClient Setup:**

```typescript
// app.module.ts
import { HttpClientModule } from '@angular/common/http';

@NgModule({
  imports: [HttpClientModule],
  // ...
})
export class AppModule {}
```

**Basic HTTP Operations:**

```typescript
@Injectable({
  providedIn: 'root',
})
export class ApiService {
  private baseUrl = 'https://api.example.com';

  constructor(private http: HttpClient) {}

  // GET request
  getUsers(): Observable<User[]> {
    return this.http.get<User[]>(`${this.baseUrl}/users`);
  }

  // GET with parameters
  getUsersWithParams(page: number, limit: number): Observable<User[]> {
    const params = new HttpParams().set('page', page.toString()).set('limit', limit.toString());

    return this.http.get<User[]>(`${this.baseUrl}/users`, { params });
  }

  // GET with headers
  getUser(id: string): Observable<User> {
    const headers = new HttpHeaders().set('Authorization', 'Bearer token');

    return this.http.get<User>(`${this.baseUrl}/users/${id}`, { headers });
  }

  // POST request
  createUser(user: User): Observable<User> {
    return this.http.post<User>(`${this.baseUrl}/users`, user);
  }

  // PUT request
  updateUser(id: string, user: User): Observable<User> {
    return this.http.put<User>(`${this.baseUrl}/users/${id}`, user);
  }

  // PATCH request
  partialUpdateUser(id: string, updates: Partial<User>): Observable<User> {
    return this.http.patch<User>(`${this.baseUrl}/users/${id}`, updates);
  }

  // DELETE request
  deleteUser(id: string): Observable<void> {
    return this.http.delete<void>(`${this.baseUrl}/users/${id}`);
  }
}
```

**Error Handling:**

```typescript
@Injectable({
  providedIn: 'root',
})
export class ApiService {
  constructor(private http: HttpClient) {}

  getUsers(): Observable<User[]> {
    return this.http.get<User[]>(`${this.baseUrl}/users`).pipe(catchError(this.handleError));
  }

  private handleError(error: HttpErrorResponse) {
    let errorMessage = '';

    if (error.error instanceof ErrorEvent) {
      // Client-side error
      errorMessage = `Error: ${error.error.message}`;
    } else {
      // Server-side error
      errorMessage = `Error Code: ${error.status}\nMessage: ${error.message}`;
    }

    console.error(errorMessage);
    return throwError(() => new Error(errorMessage));
  }

  // Specific error handling
  getUser(id: string): Observable<User> {
    return this.http.get<User>(`${this.baseUrl}/users/${id}`).pipe(
      catchError((error) => {
        if (error.status === 404) {
          return of(null); // Return null for not found
        }
        return throwError(() => error);
      }),
    );
  }
}
```

**Request and Response Interceptors:**

```typescript
// Auth interceptor
@Injectable()
export class AuthInterceptor implements HttpInterceptor {
  constructor(private authService: AuthService) {}

  intercept(req: HttpRequest<any>, next: HttpHandler): Observable<HttpEvent<any>> {
    // Add auth token
    const authToken = this.authService.getAuthToken();
    const authReq = req.clone({
      headers: req.headers.set('Authorization', `Bearer ${authToken}`),
    });

    return next.handle(authReq);
  }
}

// Logging interceptor
@Injectable()
export class LoggingInterceptor implements HttpInterceptor {
  intercept(req: HttpRequest<any>, next: HttpHandler): Observable<HttpEvent<any>> {
    console.log('HTTP Request:', req.method, req.url);

    return next.handle(req).pipe(
      tap((event) => {
        if (event instanceof HttpResponse) {
          console.log('HTTP Response:', event.status, event.body);
        }
      }),
      catchError((error) => {
        console.error('HTTP Error:', error);
        return throwError(() => error);
      }),
    );
  }
}

// Error handling interceptor
@Injectable()
export class ErrorInterceptor implements HttpInterceptor {
  constructor(private notificationService: NotificationService) {}

  intercept(req: HttpRequest<any>, next: HttpHandler): Observable<HttpEvent<any>> {
    return next.handle(req).pipe(
      catchError((error) => {
        if (error.status === 401) {
          // Handle unauthorized access
          this.notificationService.showError('Please log in to continue');
          // Redirect to login
        } else if (error.status === 500) {
          this.notificationService.showError('Server error occurred');
        }

        return throwError(() => error);
      }),
    );
  }
}
```

**Progress Tracking:**

```typescript
export class FileUploadService {
  constructor(private http: HttpClient) {}

  uploadFile(file: File): Observable<HttpEvent<any>> {
    const formData = new FormData();
    formData.append('file', file);

    return this.http.post(`${this.baseUrl}/upload`, formData, {
      reportProgress: true,
      observe: 'events',
    });
  }

  // Usage in component
  uploadFile(file: File) {
    this.fileUploadService.uploadFile(file).subscribe((event) => {
      if (event.type === HttpEventType.UploadProgress) {
        const progress = Math.round((100 * event.loaded) / event.total);
        console.log(`Upload progress: ${progress}%`);
      } else if (event.type === HttpEventType.Response) {
        console.log('Upload complete:', event.body);
      }
    });
  }
}
```

**Caching Strategies:**

```typescript
@Injectable({
  providedIn: 'root',
})
export class CacheService {
  private cache = new Map<string, { data: any; timestamp: number }>();
  private cacheTimeout = 5 * 60 * 1000; // 5 minutes

  getCachedData<T>(key: string, fetchFn: () => Observable<T>): Observable<T> {
    const cached = this.cache.get(key);

    if (cached && Date.now() - cached.timestamp < this.cacheTimeout) {
      return of(cached.data);
    }

    return fetchFn().pipe(
      tap((data) => {
        this.cache.set(key, { data, timestamp: Date.now() });
      }),
    );
  }

  clearCache() {
    this.cache.clear();
  }
}

// Usage
export class ApiService {
  constructor(
    private http: HttpClient,
    private cacheService: CacheService,
  ) {}

  getUsers(): Observable<User[]> {
    return this.cacheService.getCachedData('users', () =>
      this.http.get<User[]>(`${this.baseUrl}/users`),
    );
  }
}
```

**Retry Logic:**

```typescript
export class ApiService {
  constructor(private http: HttpClient) {}

  getUsers(): Observable<User[]> {
    return this.http.get<User[]>(`${this.baseUrl}/users`).pipe(
      retry(3), // Retry 3 times
      delay(1000), // Wait 1 second between retries
      catchError((error) => {
        console.error('Failed after 3 retries:', error);
        return throwError(() => error);
      }),
    );
  }

  // Conditional retry
  getUsersWithConditionalRetry(): Observable<User[]> {
    return this.http.get<User[]>(`${this.baseUrl}/users`).pipe(
      retryWhen((errors) =>
        errors.pipe(
          mergeMap((error, index) => {
            if (index < 3 && error.status === 503) {
              return of(error).pipe(delay(1000 * (index + 1)));
            }
            return throwError(() => error);
          }),
        ),
      ),
    );
  }
}
```

**Request/Response Transformation:**

```typescript
// Request interceptor for transformation
@Injectable()
export class TransformInterceptor implements HttpInterceptor {
  intercept(req: HttpRequest<any>, next: HttpHandler): Observable<HttpEvent<any>> {
    // Transform request
    if (req.method === 'POST' || req.method === 'PUT') {
      const transformedBody = this.transformRequest(req.body);
      const newReq = req.clone({ body: transformedBody });
      return next.handle(newReq);
    }

    return next.handle(req);
  }

  private transformRequest(body: any): any {
    // Add timestamp
    return {
      ...body,
      _timestamp: new Date().toISOString(),
    };
  }
}

// Response interceptor for transformation
@Injectable()
export class ResponseTransformInterceptor implements HttpInterceptor {
  intercept(req: HttpRequest<any>, next: HttpHandler): Observable<HttpEvent<any>> {
    return next.handle(req).pipe(
      map((event) => {
        if (event instanceof HttpResponse) {
          const transformedBody = this.transformResponse(event.body);
          return event.clone({ body: transformedBody });
        }
        return event;
      }),
    );
  }

  private transformResponse(body: any): any {
    // Convert date strings to Date objects
    if (body && typeof body === 'object') {
      return this.convertDates(body);
    }
    return body;
  }

  private convertDates(obj: any): any {
    if (obj instanceof Date) {
      return obj;
    }
    if (obj && typeof obj === 'object') {
      Object.keys(obj).forEach((key) => {
        if (typeof obj[key] === 'string' && this.isDateString(obj[key])) {
          obj[key] = new Date(obj[key]);
        } else if (typeof obj[key] === 'object') {
          obj[key] = this.convertDates(obj[key]);
        }
      });
    }
    return obj;
  }

  private isDateString(value: string): boolean {
    return /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/.test(value);
  }
}
```

**Best Practices:**

- Use interceptors for cross-cutting concerns
- Implement proper error handling and user feedback
- Use caching for performance optimization
- Implement retry logic for unreliable networks
- Handle authentication and authorization properly
- Use TypeScript interfaces for type safety
- Consider using async/await for simpler async code
- Implement proper loading states and user feedback

### Q25: What are Angular services and how do you create HTTP services?

**Answer:**
Angular services are singleton classes that handle business logic, data management, and external communications. HTTP services specifically handle communication with REST APIs.

**Creating Basic Services:**

```typescript
// user.service.ts
@Injectable({
  providedIn: 'root',
})
export class UserService {
  private apiUrl = 'https://api.example.com/users';
  private usersSubject = new BehaviorSubject<User[]>([]);
  public users$ = this.usersSubject.asObservable();

  constructor(private http: HttpClient) {}

  // Get all users
  getUsers(): Observable<User[]> {
    return this.http.get<User[]>(this.apiUrl).pipe(
      tap((users) => this.usersSubject.next(users)),
      catchError(this.handleError),
    );
  }

  // Get user by ID
  getUser(id: string): Observable<User> {
    return this.http.get<User>(`${this.apiUrl}/${id}`).pipe(catchError(this.handleError));
  }

  // Create user
  createUser(user: CreateUserRequest): Observable<User> {
    return this.http.post<User>(this.apiUrl, user).pipe(
      tap((newUser) => {
        const currentUsers = this.usersSubject.value;
        this.usersSubject.next([...currentUsers, newUser]);
      }),
      catchError(this.handleError),
    );
  }

  // Update user
  updateUser(id: string, user: UpdateUserRequest): Observable<User> {
    return this.http.put<User>(`${this.apiUrl}/${id}`, user).pipe(
      tap((updatedUser) => {
        const currentUsers = this.usersSubject.value;
        const index = currentUsers.findIndex((u) => u.id === id);
        if (index !== -1) {
          currentUsers[index] = updatedUser;
          this.usersSubject.next([...currentUsers]);
        }
      }),
      catchError(this.handleError),
    );
  }

  // Delete user
  deleteUser(id: string): Observable<void> {
    return this.http.delete<void>(`${this.apiUrl}/${id}`).pipe(
      tap(() => {
        const currentUsers = this.usersSubject.value;
        this.usersSubject.next(currentUsers.filter((u) => u.id !== id));
      }),
      catchError(this.handleError),
    );
  }

  private handleError(error: HttpErrorResponse) {
    console.error('API Error:', error);
    return throwError(() => new Error('Something went wrong with the API'));
  }
}
```

**Advanced Service Patterns:**

1. **CRUD Service Base Class:**

```typescript
export abstract class ApiService<T> {
  protected abstract baseUrl: string;

  constructor(protected http: HttpClient) {}

  getAll(): Observable<T[]> {
    return this.http.get<T[]>(this.baseUrl);
  }

  getById(id: string): Observable<T> {
    return this.http.get<T>(`${this.baseUrl}/${id}`);
  }

  create(item: Partial<T>): Observable<T> {
    return this.http.post<T>(this.baseUrl, item);
  }

  update(id: string, item: Partial<T>): Observable<T> {
    return this.http.put<T>(`${this.baseUrl}/${id}`, item);
  }

  delete(id: string): Observable<void> {
    return this.http.delete<void>(`${this.baseUrl}/${id}`);
  }
}

// Usage
@Injectable({
  providedIn: 'root',
})
export class UserService extends ApiService<User> {
  protected baseUrl = 'https://api.example.com/users';
}
```

2. **Service with State Management:**

```typescript
@Injectable({
  providedIn: 'root',
})
export class ProductService {
  private productsSubject = new BehaviorSubject<Product[]>([]);
  private loadingSubject = new BehaviorSubject<boolean>(false);
  private errorSubject = new BehaviorSubject<string | null>(null);

  public products$ = this.productsSubject.asObservable();
  public loading$ = this.loadingSubject.asObservable();
  public error$ = this.errorSubject.asObservable();

  constructor(private http: HttpClient) {}

  loadProducts() {
    this.loadingSubject.next(true);
    this.errorSubject.next(null);

    this.http
      .get<Product[]>(this.apiUrl)
      .pipe(finalize(() => this.loadingSubject.next(false)))
      .subscribe({
        next: (products) => this.productsSubject.next(products),
        error: (error) => this.errorSubject.next('Failed to load products'),
      });
  }

  addProduct(product: CreateProductRequest) {
    this.http.post<Product>(this.apiUrl, product).subscribe({
      next: (newProduct) => {
        const currentProducts = this.productsSubject.value;
        this.productsSubject.next([...currentProducts, newProduct]);
      },
      error: (error) => this.errorSubject.next('Failed to add product'),
    });
  }
}
```

3. **Service with Caching:**

```typescript
@Injectable({
  providedIn: 'root',
})
export class CacheService {
  private cache = new Map<string, { data: any; timestamp: number }>();

  set(key: string, data: any, ttl: number = 300000) {
    // 5 minutes default
    this.cache.set(key, {
      data,
      timestamp: Date.now() + ttl,
    });
  }

  get<T>(key: string): T | null {
    const item = this.cache.get(key);
    if (item && item.timestamp > Date.now()) {
      return item.data as T;
    }
    this.cache.delete(key);
    return null;
  }

  clear() {
    this.cache.clear();
  }
}

@Injectable({
  providedIn: 'root',
})
export class CachedApiService {
  constructor(
    private http: HttpClient,
    private cache: CacheService,
  ) {}

  getUsers(): Observable<User[]> {
    const cachedUsers = this.cache.get<User[]>('users');
    if (cachedUsers) {
      return of(cachedUsers);
    }

    return this.http
      .get<User[]>('https://api.example.com/users')
      .pipe(tap((users) => this.cache.set('users', users)));
  }
}
```

4. **Service with Retry Logic:**

```typescript
@Injectable({
  providedIn: 'root',
})
export class RobustApiService {
  constructor(private http: HttpClient) {}

  getUsers(): Observable<User[]> {
    return this.http.get<User[]>('https://api.example.com/users').pipe(
      retry(3),
      delay(1000),
      catchError((error) => {
        console.error('API call failed after retries:', error);
        return throwError(() => error);
      }),
    );
  }

  // Conditional retry based on error type
  getDataWithSmartRetry<T>(url: string): Observable<T> {
    return this.http.get<T>(url).pipe(
      retryWhen((errors) =>
        errors.pipe(
          mergeMap((error, index) => {
            if (index < 3 && error.status >= 500) {
              return of(error).pipe(delay(1000 * Math.pow(2, index)));
            }
            return throwError(() => error);
          }),
        ),
      ),
    );
  }
}
```

5. **Service with Request/Response Transformation:**

```typescript
@Injectable({
  providedIn: 'root',
})
export class TransformingApiService {
  constructor(private http: HttpClient) {}

  getUsers(): Observable<User[]> {
    return this.http
      .get<any[]>('https://api.example.com/users')
      .pipe(map((users) => users.map((user) => this.transformUser(user))));
  }

  private transformUser(apiUser: any): User {
    return {
      id: apiUser.user_id,
      name: apiUser.full_name,
      email: apiUser.email_address,
      createdAt: new Date(apiUser.created_at),
      isActive: apiUser.status === 'active',
    };
  }

  createUser(user: CreateUserRequest): Observable<User> {
    const apiUser = this.transformToApiUser(user);
    return this.http
      .post<any>('https://api.example.com/users', apiUser)
      .pipe(map((apiUser) => this.transformUser(apiUser)));
  }

  private transformToApiUser(user: CreateUserRequest): any {
    return {
      full_name: user.name,
      email_address: user.email,
      status: 'active',
    };
  }
}
```

**Service Testing:**

```typescript
describe('UserService', () => {
  let service: UserService;
  let httpMock: HttpTestingController;
  let mockUsers: User[];

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule],
      providers: [UserService],
    });

    service = TestBed.inject(UserService);
    httpMock = TestBed.inject(HttpTestingController);
    mockUsers = [
      { id: '1', name: 'John Doe', email: 'john@example.com' },
      { id: '2', name: 'Jane Smith', email: 'jane@example.com' },
    ];
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should get users', () => {
    service.getUsers().subscribe((users) => {
      expect(users).toEqual(mockUsers);
    });

    const req = httpMock.expectOne('https://api.example.com/users');
    expect(req.request.method).toBe('GET');
    req.flush(mockUsers);
  });

  it('should handle errors', () => {
    const errorMessage = 'Server error';

    service.getUsers().subscribe({
      next: () => fail('should have failed with 500 error'),
      error: (error) => {
        expect(error.message).toContain('Something went wrong with the API');
      },
    });

    const req = httpMock.expectOne('https://api.example.com/users');
    req.flush(errorMessage, { status: 500, statusText: 'Server Error' });
  });
});
```

**Service Best Practices:**

1. **Use providedIn: 'root'** for singleton services
2. **Implement proper error handling** with user-friendly messages
3. **Use RxJS operators** for data transformation and manipulation
4. **Implement caching** for performance optimization
5. **Use interceptors** for cross-cutting concerns
6. **Follow naming conventions** (e.g., UserService, not User)
7. **Keep services focused** on specific responsibilities
8. **Use TypeScript interfaces** for type safety
9. **Implement proper cleanup** in ngOnDestroy when needed
10. **Write unit tests** for service logic

**Service Architecture Patterns:**

1. **Repository Pattern:**

```typescript
@Injectable({
  providedIn: 'root',
})
export class UserRepository {
  constructor(private apiService: ApiService) {}

  async getUser(id: string): Promise<User> {
    // Try cache first, then API
    return this.apiService.getUser(id).toPromise();
  }
}
```

2. **Factory Pattern:**

```typescript
@Injectable({
  providedIn: 'root',
})
export class ServiceFactory {
  constructor(
    private userService: UserService,
    private productService: ProductService,
  ) {}

  getService<T>(type: 'user' | 'product'): T {
    switch (type) {
      case 'user':
        return this.userService as T;
      case 'product':
        return this.productService as T;
      default:
        throw new Error('Unknown service type');
    }
  }
}
```

3. **Strategy Pattern:**

```typescript
@Injectable({
  providedIn: 'root',
})
export class DataSyncService {
  private strategies: Map<string, SyncStrategy> = new Map();

  registerStrategy(name: string, strategy: SyncStrategy) {
    this.strategies.set(name, strategy);
  }

  syncData(strategyName: string, data: any) {
    const strategy = this.strategies.get(strategyName);
    if (!strategy) {
      throw new Error(`Strategy ${strategyName} not found`);
    }
    return strategy.sync(data);
  }
}
```

**Best Practices Summary:**

- Use dependency injection for service dependencies
- Implement proper error handling and logging
- Use RxJS for reactive programming patterns
- Consider caching strategies for performance
- Write comprehensive unit tests
- Follow consistent naming and structure patterns
- Use TypeScript for type safety and better development experience
- Document service interfaces and usage patterns

---

## 7. Forms & Validation

### Q26: Explain template-driven forms vs reactive forms in Angular

**Answer:**
Angular provides two approaches for handling forms: Template-Driven Forms and Reactive Forms, each with different philosophies and use cases.

**Template-Driven Forms:**

Template-driven forms are simpler to set up and rely heavily on directives in the template to create and manage the form control hierarchy.

**Setup:**

```typescript
// app.module.ts
import { FormsModule } from '@angular/forms';

@NgModule({
  imports: [FormsModule],
})
export class AppModule {}
```

**Basic Template-Driven Form:**

```typescript
// Component
export class UserFormComponent {
  user = {
    name: '',
    email: '',
    password: '',
  };

  onSubmit(form: NgForm) {
    if (form.valid) {
      console.log('Form submitted:', this.user);
    }
  }
}
```

```html
<!-- Template -->
<form #userForm="ngForm" (ngSubmit)="onSubmit(userForm)">
  <div>
    <label for="name">Name:</label>
    <input id="name" name="name" [(ngModel)]="user.name" required />
    <div *ngIf="userForm.controls.name?.invalid && userForm.controls.name?.touched">
      Name is required
    </div>
  </div>

  <div>
    <label for="email">Email:</label>
    <input id="email" name="email" type="email" [(ngModel)]="user.email" required email />
    <div *ngIf="userForm.controls.email?.invalid && userForm.controls.email?.touched">
      Please enter a valid email
    </div>
  </div>

  <button type="submit" [disabled]="userForm.invalid">Submit</button>
</form>
```

**Reactive Forms:**

Reactive forms provide more control and are created programmatically in the component class, making them more testable and flexible.

**Setup:**

```typescript
// app.module.ts
import { ReactiveFormsModule } from '@angular/forms';

@NgModule({
  imports: [ReactiveFormsModule],
})
export class AppModule {}
```

**Basic Reactive Form:**

```typescript
// Component
import { FormBuilder, FormGroup, Validators } from '@angular/forms';

export class UserFormComponent implements OnInit {
  userForm: FormGroup;

  constructor(private fb: FormBuilder) {}

  ngOnInit() {
    this.userForm = this.fb.group({
      name: ['', [Validators.required, Validators.minLength(2)]],
      email: ['', [Validators.required, Validators.email]],
      password: ['', [Validators.required, Validators.minLength(6)]],
    });
  }

  onSubmit() {
    if (this.userForm.valid) {
      console.log('Form submitted:', this.userForm.value);
    }
  }

  get name() {
    return this.userForm.get('name');
  }
  get email() {
    return this.userForm.get('email');
  }
  get password() {
    return this.userForm.get('password');
  }
}
```

```html
<!-- Template -->
<form [formGroup]="userForm" (ngSubmit)="onSubmit()">
  <div>
    <label for="name">Name:</label>
    <input id="name" formControlName="name" />
    <div *ngIf="name?.invalid && name?.touched">
      Name is required and must be at least 2 characters
    </div>
  </div>

  <div>
    <label for="email">Email:</label>
    <input id="email" formControlName="email" type="email" />
    <div *ngIf="email?.invalid && email?.touched">Please enter a valid email</div>
  </div>

  <button type="submit" [disabled]="userForm.invalid">Submit</button>
</form>
```

**Advanced Reactive Forms:**

1. **Dynamic Form Controls:**

```typescript
export class DynamicFormComponent implements OnInit {
  form: FormGroup;

  constructor(private fb: FormBuilder) {}

  ngOnInit() {
    this.form = this.fb.group({
      items: this.fb.array([this.createItem()]),
    });
  }

  createItem(): FormGroup {
    return this.fb.group({
      name: ['', Validators.required],
      quantity: [1, [Validators.required, Validators.min(1)]],
      price: [0, Validators.required],
    });
  }

  get items(): FormArray {
    return this.form.get('items') as FormArray;
  }

  addItem() {
    this.items.push(this.createItem());
  }

  removeItem(index: number) {
    this.items.removeAt(index);
  }
}
```

2. **Custom Validators:**

```typescript
// Custom validator function
export function forbiddenNameValidator(forbiddenName: string): ValidatorFn {
  return (control: AbstractControl): ValidationErrors | null => {
    const forbidden = new RegExp(forbiddenName, 'i').test(control.value);
    return forbidden ? { forbiddenName: { value: control.value } } : null;
  };
}

// Async validator
export function uniqueEmailValidator(userService: UserService): AsyncValidatorFn {
  return (control: AbstractControl): Observable<ValidationErrors | null> => {
    return userService.checkEmailExists(control.value).pipe(
      map((exists) => (exists ? { emailExists: true } : null)),
      catchError(() => of(null)),
    );
  };
}

// Usage in component
this.userForm = this.fb.group({
  name: ['', [Validators.required, forbiddenNameValidator('admin')]],
  email: ['', [Validators.required, Validators.email], [uniqueEmailValidator(this.userService)]],
});
```

3. **Conditional Validation:**

```typescript
export class ConditionalFormComponent implements OnInit {
  form: FormGroup;

  constructor(private fb: FormBuilder) {}

  ngOnInit() {
    this.form = this.fb.group({
      userType: ['user', Validators.required],
      companyName: [''],
      personalId: [''],
    });

    this.form.get('userType')?.valueChanges.subscribe((userType) => {
      const companyNameControl = this.form.get('companyName');
      const personalIdControl = this.form.get('personalId');

      if (userType === 'company') {
        companyNameControl?.setValidators([Validators.required]);
        personalIdControl?.clearValidators();
      } else {
        companyNameControl?.clearValidators();
        personalIdControl?.setValidators([Validators.required]);
      }

      companyNameControl?.updateValueAndValidity();
      personalIdControl?.updateValueAndValidity();
    });
  }
}
```

**Template-Driven vs Reactive Forms Comparison:**

| Aspect             | Template-Driven          | Reactive               |
| ------------------ | ------------------------ | ---------------------- |
| **Setup**          | Simple, less code        | More setup, explicit   |
| **Testing**        | Harder to test           | Easier to test         |
| **Validation**     | Template-based           | Code-based             |
| **Dynamic Forms**  | Limited                  | Full support           |
| **Performance**    | Slower for complex forms | Better performance     |
| **Flexibility**    | Less flexible            | Highly flexible        |
| **Learning Curve** | Easier for beginners     | Steeper learning curve |

**When to Use Each:**

**Template-Driven Forms:**

- Simple forms with basic validation
- Rapid prototyping
- Forms with minimal dynamic behavior
- When you prefer template-centric approach

**Reactive Forms:**

- Complex forms with dynamic behavior
- Forms requiring custom validation
- When you need better testability
- Forms with conditional logic
- When you prefer code-centric approach

### Q27: How do you implement custom form validation in Angular?

**Answer:**
Angular provides multiple ways to implement custom form validation, from simple validator functions to complex async validators.

**1. Custom Synchronous Validators:**

```typescript
// Custom validator function
export function passwordStrengthValidator(): ValidatorFn {
  return (control: AbstractControl): ValidationErrors | null => {
    const value = control.value;

    if (!value) {
      return null;
    }

    const hasUpperCase = /[A-Z]/.test(value);
    const hasLowerCase = /[a-z]/.test(value);
    const hasNumeric = /[0-9]/.test(value);
    const hasSpecialChar = /[!@#$%^&*(),.?":{}|<>]/.test(value);

    const passwordValid = hasUpperCase && hasLowerCase && hasNumeric && hasSpecialChar;

    return !passwordValid ? { passwordStrength: true } : null;
  };
}

// Usage
this.form = this.fb.group({
  password: ['', [Validators.required, passwordStrengthValidator()]],
});
```

**2. Cross-Field Validation:**

```typescript
// Cross-field validator
export function passwordMatchValidator(group: AbstractControl): ValidationErrors | null {
  const password = group.get('password')?.value;
  const confirmPassword = group.get('confirmPassword')?.value;

  return password === confirmPassword ? null : { passwordsMismatch: true };
}

// Usage
this.form = this.fb.group(
  {
    password: ['', [Validators.required, Validators.minLength(8)]],
    confirmPassword: ['', Validators.required],
  },
  { validators: passwordMatchValidator },
);
```

**3. Custom Async Validators:**

```typescript
@Injectable({ providedIn: 'root' })
export class UniqueUsernameValidator implements AsyncValidator {
  constructor(private userService: UserService) {}

  validate(control: AbstractControl): Observable<ValidationErrors | null> {
    return this.userService.checkUsernameExists(control.value).pipe(
      map((exists) => (exists ? { usernameExists: true } : null)),
      catchError(() => of(null)), // Return null on error to avoid blocking form
    );
  }
}

// Usage
this.form = this.fb.group({
  username: [
    '',
    [Validators.required],
    [this.uniqueUsernameValidator.validate.bind(this.uniqueUsernameValidator)],
  ],
});
```

**4. Conditional Validators:**

```typescript
export class ConditionalValidator {
  static requiredIf(condition: () => boolean): ValidatorFn {
    return (control: AbstractControl): ValidationErrors | null => {
      if (condition() && !control.value) {
        return { required: true };
      }
      return null;
    };
  }

  static minLengthIf(minLength: number, condition: () => boolean): ValidatorFn {
    return (control: AbstractControl): ValidationErrors | null => {
      if (condition() && control.value && control.value.length < minLength) {
        return { minlength: { requiredLength: minLength, actualLength: control.value.length } };
      }
      return null;
    };
  }
}

// Usage
this.form = this.fb.group({
  userType: ['user'],
  companyName: [
    '',
    [
      ConditionalValidator.requiredIf(() => this.form?.get('userType')?.value === 'company'),
      ConditionalValidator.minLengthIf(3, () => this.form?.get('userType')?.value === 'company'),
    ],
  ],
});
```

**5. Custom Validator Directive:**

```typescript
@Directive({
  selector: '[appCustomValidator]',
  providers: [{ provide: NG_VALIDATORS, useExisting: CustomValidatorDirective, multi: true }]
})
export class CustomValidatorDirective implements Validator {
  @Input('appCustomValidator') validatorType: string;

  validate(control: AbstractControl): ValidationErrors | null {
    if (!control.value) {
      return null;
    }

    switch (this.validatorType) {
      case 'email':
        return this.validateEmail(control);
      case 'phone':
        return this.validatePhone(control);
      case 'postalCode':
        return this.validatePostalCode(control);
      default:
        return null;
    }
  }

  private validateEmail(control: AbstractControl): ValidationErrors | null {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(control.value) ? null : { invalidEmail: true };
  }

  private validatePhone(control: AbstractControl): ValidationErrors | null {
    const phoneRegex = /^\+?[\d\s\-\(\)]{10,}$/;
    return phoneRegex.test(control.value) ? null : { invalidPhone: true };
  }

  private validatePostalCode(control: AbstractControl): ValidationErrors | null {
    const postalCodeRegex = /^[A-Za-z]\d[A-Za-z][ -]?\d[A-Za-z]\d$/;
    return postalCodeRegex.test(control.value) ? null : { invalidPostalCode: true };
  }
}

// Usage in template
<input appCustomValidator="email" formControlName="email">
```

**6. Dynamic Validation:**

```typescript
export class DynamicValidationComponent implements OnInit {
  form: FormGroup;

  constructor(private fb: FormBuilder) {}

  ngOnInit() {
    this.form = this.fb.group({
      country: [''],
      postalCode: [''],
    });

    this.setupDynamicValidation();
  }

  setupDynamicValidation() {
    this.form.get('country')?.valueChanges.subscribe((country) => {
      const postalCodeControl = this.form.get('postalCode');

      if (country === 'US') {
        postalCodeControl?.setValidators([
          Validators.required,
          Validators.pattern(/^\d{5}(-\d{4})?$/),
        ]);
      } else if (country === 'CA') {
        postalCodeControl?.setValidators([
          Validators.required,
          Validators.pattern(/^[A-Za-z]\d[A-Za-z][ -]?\d[A-Za-z]\d$/),
        ]);
      } else {
        postalCodeControl?.clearValidators();
      }

      postalCodeControl?.updateValueAndValidity();
    });
  }
}
```

**7. Validation with Custom Error Messages:**

```typescript
export class ValidationMessagesComponent {
  static getValidationMessage(control: AbstractControl, controlName: string): string {
    if (control?.errors) {
      if (control.errors['required']) {
        return `${controlName} is required.`;
      }
      if (control.errors['minlength']) {
        const requiredLength = control.errors['minlength'].requiredLength;
        return `${controlName} must be at least ${requiredLength} characters long.`;
      }
      if (control.errors['email']) {
        return 'Please enter a valid email address.';
      }
      if (control.errors['passwordStrength']) {
        return 'Password must contain uppercase, lowercase, number, and special character.';
      }
      if (control.errors['usernameExists']) {
        return 'This username is already taken.';
      }
    }
    return '';
  }
}

// Usage in template
<div *ngIf="form.get('password')?.invalid && form.get('password')?.touched">
  {{ ValidationMessagesComponent.getValidationMessage(form.get('password'), 'Password') }}
</div>
```

**8. Complex Business Logic Validation:**

```typescript
export function businessRuleValidator(): ValidatorFn {
  return (control: AbstractControl): ValidationErrors | null => {
    if (!control.value) {
      return null;
    }

    const age = control.value.age;
    const employmentStatus = control.value.employmentStatus;
    const income = control.value.income;

    // Business rule: If unemployed, income must be 0
    if (employmentStatus === 'unemployed' && income > 0) {
      return { invalidIncomeForUnemployed: true };
    }

    // Business rule: If under 18, cannot have full-time employment
    if (age < 18 && employmentStatus === 'full-time') {
      return { underageFullTime: true };
    }

    return null;
  };
}

// Usage with form group
this.form = this.fb.group(
  {
    age: ['', Validators.required],
    employmentStatus: ['', Validators.required],
    income: ['', Validators.required],
  },
  { validators: businessRuleValidator() },
);
```

**Best Practices for Custom Validation:**

1. **Keep validators pure functions** - no side effects
2. **Use descriptive error keys** for better error handling
3. **Handle async validation errors gracefully**
4. **Provide clear error messages** to users
5. **Use validation groups** for complex forms
6. **Test validators independently**
7. **Consider performance** for complex validation logic
8. **Use reactive forms** for complex validation scenarios

### Q28: How do you handle form arrays and dynamic forms in Angular?

**Answer:**
Angular's FormArray allows you to handle dynamic lists of form controls that can be added or removed at runtime.

**Basic FormArray Implementation:**

```typescript
export class DynamicFormComponent implements OnInit {
  userForm: FormGroup;

  constructor(private fb: FormBuilder) {}

  ngOnInit() {
    this.userForm = this.fb.group({
      name: ['', Validators.required],
      email: ['', [Validators.required, Validators.email]],
      skills: this.fb.array([]),
    });
  }

  get skills(): FormArray {
    return this.userForm.get('skills') as FormArray;
  }

  addSkill() {
    const skillGroup = this.fb.group({
      name: ['', Validators.required],
      level: ['', Validators.required],
      years: ['', [Validators.required, Validators.min(0)]],
    });
    this.skills.push(skillGroup);
  }

  removeSkill(index: number) {
    this.skills.removeAt(index);
  }

  onSubmit() {
    if (this.userForm.valid) {
      console.log('Form data:', this.userForm.value);
    }
  }
}
```

```html
<!-- Template -->
<form [formGroup]="userForm" (ngSubmit)="onSubmit()">
  <div>
    <label>Name:</label>
    <input formControlName="name" />
  </div>

  <div>
    <label>Email:</label>
    <input formControlName="email" type="email" />
  </div>

  <div formArrayName="skills">
    <h3>Skills</h3>
    <div *ngFor="let skill of skills.controls; let i = index" [formGroupName]="i">
      <div>
        <label>Skill Name:</label>
        <input formControlName="name" />
      </div>
      <div>
        <label>Level:</label>
        <select formControlName="level">
          <option value="beginner">Beginner</option>
          <option value="intermediate">Intermediate</option>
          <option value="expert">Expert</option>
        </select>
      </div>
      <div>
        <label>Years of Experience:</label>
        <input formControlName="years" type="number" />
      </div>
      <button type="button" (click)="removeSkill(i)">Remove</button>
    </div>
  </div>

  <button type="button" (click)="addSkill()">Add Skill</button>
  <button type="submit">Submit</button>
</form>
```

**Advanced FormArray Patterns:**

1. **Nested FormArrays:**

```typescript
export class ComplexFormComponent implements OnInit {
  form: FormGroup;

  ngOnInit() {
    this.form = this.fb.group({
      project: this.fb.group({
        name: ['', Validators.required],
        team: this.fb.array([]),
      }),
    });
  }

  get team(): FormArray {
    return this.form.get('project.team') as FormArray;
  }

  addTeamMember() {
    const memberGroup = this.fb.group({
      name: ['', Validators.required],
      skills: this.fb.array([this.fb.control('', Validators.required)]),
      assignments: this.fb.array([]),
    });
    this.team.push(memberGroup);
  }

  addSkillToMember(memberIndex: number) {
    const skills = this.team.at(memberIndex).get('skills') as FormArray;
    skills.push(this.fb.control('', Validators.required));
  }

  addAssignmentToMember(memberIndex: number) {
    const assignments = this.team.at(memberIndex).get('assignments') as FormArray;
    const assignmentGroup = this.fb.group({
      task: ['', Validators.required],
      deadline: ['', Validators.required],
      priority: ['medium', Validators.required],
    });
    assignments.push(assignmentGroup);
  }
}
```

2. **Dynamic FormArray with Predefined Options:**

```typescript
export class SurveyFormComponent implements OnInit {
  form: FormGroup;
  questionTypes = [
    { value: 'text', label: 'Text Input' },
    { value: 'number', label: 'Number Input' },
    { value: 'select', label: 'Dropdown' },
    { value: 'checkbox', label: 'Checkbox' },
  ];

  ngOnInit() {
    this.form = this.fb.group({
      surveyName: ['', Validators.required],
      questions: this.fb.array([]),
    });
  }

  get questions(): FormArray {
    return this.form.get('questions') as FormArray;
  }

  addQuestion() {
    const questionGroup = this.fb.group({
      text: ['', Validators.required],
      type: ['text', Validators.required],
      required: [false],
      options: this.fb.array([]), // For select/checkbox types
    });

    this.questions.push(questionGroup);
  }

  addOption(questionIndex: number) {
    const options = this.questions.at(questionIndex).get('options') as FormArray;
    options.push(this.fb.control('', Validators.required));
  }

  removeOption(questionIndex: number, optionIndex: number) {
    const options = this.questions.at(questionIndex).get('options') as FormArray;
    options.removeAt(optionIndex);
  }

  onQuestionTypeChange(questionIndex: number, type: string) {
    const question = this.questions.at(questionIndex);
    const options = question.get('options') as FormArray;

    // Clear existing options
    while (options.length !== 0) {
      options.removeAt(0);
    }

    // Add default options for select/checkbox
    if (type === 'select' || type === 'checkbox') {
      this.addOption(questionIndex);
      this.addOption(questionIndex);
    }
  }
}
```

3. **FormArray with Conditional Validation:**

```typescript
export class ConditionalArrayFormComponent implements OnInit {
  form: FormGroup;

  ngOnInit() {
    this.form = this.fb.group({
      items: this.fb.array([], Validators.minLength(1)),
    });
  }

  get items(): FormArray {
    return this.form.get('items') as FormArray;
  }

  addItem() {
    const itemGroup = this.fb.group({
      type: ['product', Validators.required],
      name: ['', Validators.required],
      quantity: [1, [Validators.required, Validators.min(1)]],
      price: [0, Validators.required],
      description: [''],
    });

    // Conditional validation based on type
    itemGroup
      .get('description')
      ?.setValidators(this.getDescriptionValidators(itemGroup.get('type')?.value));

    this.items.push(itemGroup);
  }

  private getDescriptionValidators(type: string): ValidatorFn[] {
    const validators = [Validators.required];

    if (type === 'service') {
      validators.push(Validators.minLength(50));
    }

    return validators;
  }

  onItemTypeChange(index: number) {
    const item = this.items.at(index);
    const type = item.get('type')?.value;
    const description = item.get('description');

    description?.setValidators(this.getDescriptionValidators(type));
    description?.updateValueAndValidity();
  }
}
```

4. **FormArray with Drag and Drop Reordering:**

```typescript
import { CdkDragDrop, moveItemInArray } from '@angular/cdk/drag-drop';

export class DragDropFormComponent implements OnInit {
  form: FormGroup;

  ngOnInit() {
    this.form = this.fb.group({
      steps: this.fb.array([]),
    });
  }

  get steps(): FormArray {
    return this.form.get('steps') as FormArray;
  }

  addStep() {
    const stepGroup = this.fb.group({
      title: ['', Validators.required],
      description: ['', Validators.required],
      duration: [0, Validators.required],
    });
    this.steps.push(stepGroup);
  }

  drop(event: CdkDragDrop<string[]>) {
    moveItemInArray(this.steps.controls, event.previousIndex, event.currentIndex);
    this.steps.updateValueAndValidity();
  }

  removeStep(index: number) {
    this.steps.removeAt(index);
  }
}
```

```html
<!-- Template with drag and drop -->
<div formArrayName="steps" cdkDropList (cdkDropListDropped)="drop($event)">
  <div *ngFor="let step of steps.controls; let i = index" [formGroupName]="i" cdkDrag>
    <div>
      <label>Step {{ i + 1 }} Title:</label>
      <input formControlName="title" />
    </div>
    <div>
      <label>Description:</label>
      <textarea formControlName="description"></textarea>
    </div>
    <div>
      <label>Duration (minutes):</label>
      <input formControlName="duration" type="number" />
    </div>
    <button (click)="removeStep(i)">Remove Step</button>
    <span cdkDragHandle>↕ Drag to reorder</span>
  </div>
</div>

<button (click)="addStep()">Add Step</button>
```

**FormArray Best Practices:**

1. **Use trackBy for Performance:**

```typescript
// In component
trackByFn(index: number, item: any): number {
  return index; // or item.id if available
}

// In template
<div *ngFor="let item of items.controls; trackBy: trackByFn">
```

2. **Handle FormArray Validation:**

```typescript
// Custom FormArray validator
const uniqueItemsValidator: ValidatorFn = (formArray: FormArray): ValidationErrors | null => {
  const items = formArray.controls;
  const duplicates = items.some((item, index) => {
    return items.some((otherItem, otherIndex) => {
      return index !== otherIndex && item.value.name === otherItem.value.name;
    });
  });

  return duplicates ? { duplicateItems: true } : null;
};

// Usage
this.form = this.fb.group({
  items: this.fb.array([], uniqueItemsValidator),
});
```

3. **Dynamic FormArray Initialization:**

```typescript
// Initialize with existing data
ngOnInit() {
  this.form = this.fb.group({
    items: this.fb.array([])
  });

  // Load existing data
  this.loadExistingData();
}

private loadExistingData() {
  const existingItems = this.dataService.getExistingItems();
  existingItems.forEach(item => {
    this.addItem(item);
  });
}

private addItem(data?: any) {
  const itemGroup = this.fb.group({
    name: [data?.name || '', Validators.required],
    quantity: [data?.quantity || 1, Validators.required],
    price: [data?.price || 0, Validators.required]
  });
  this.items.push(itemGroup);
}
```

4. **FormArray Cleanup:**

```typescript
ngOnDestroy() {
  // Remove all items when component is destroyed
  while (this.items.length !== 0) {
    this.items.removeAt(0);
  }
}
```

**Performance Considerations:**

1. **Use OnPush Change Detection:**

```typescript
@Component({
  selector: 'app-dynamic-form',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class DynamicFormComponent {}
```

2. **Optimize Large FormArrays:**

```typescript
// Virtualization for large lists
import { ScrollingModule } from '@angular/cdk/scrolling';

// Use virtualization for long lists
<cdk-virtual-scroll-viewport itemSize="50" class="example-viewport">
  <div *cdkVirtualFor="let item of items.controls; trackBy: trackByFn">
    <!-- Form controls -->
  </div>
</cdk-virtual-scroll-viewport>
```

3. **Debounce Validation:**

```typescript
// Debounce form validation for better performance
import { debounceTime } from 'rxjs/operators';

ngOnInit() {
  this.form.valueChanges.pipe(
    debounceTime(300)
  ).subscribe(() => {
    // Handle form changes
  });
}
```

**Best Practices Summary:**

- Use FormArray for dynamic lists of form controls
- Implement proper validation for dynamic content
- Handle cleanup in ngOnDestroy
- Use trackBy for better performance with ngFor
- Consider virtualization for large FormArrays
- Implement drag-and-drop for reordering when needed
- Use conditional validation based on form state
- Provide clear user feedback for form operations

---

## 8. Performance Optimization

### Q29: What are the key performance optimization techniques in Angular?

**Answer:**
Angular performance optimization involves multiple strategies to improve application speed, responsiveness, and user experience.

**1. Change Detection Optimization:**

```typescript
// Use OnPush change detection strategy
@Component({
  selector: 'app-user-list',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div *ngFor="let user of users$ | async">
      {{ user.name }}
    </div>
  `,
})
export class UserListComponent {
  @Input() users: User[];
  users$ = new BehaviorSubject<User[]>([]);
}
```

**2. Lazy Loading Modules:**

```typescript
// app-routing.module.ts
const routes: Routes = [
  {
    path: 'dashboard',
    loadChildren: () => import('./dashboard/dashboard.module').then((m) => m.DashboardModule),
  },
  { path: 'users', loadChildren: () => import('./users/users.module').then((m) => m.UsersModule) },
  { path: 'admin', loadChildren: () => import('./admin/admin.module').then((m) => m.AdminModule) },
];

@NgModule({
  imports: [
    RouterModule.forRoot(routes, {
      preloadingStrategy: PreloadAllModules, // Preload all lazy modules
    }),
  ],
})
export class AppRoutingModule {}
```

**3. Virtualization for Long Lists:**

```typescript
// Install @angular/cdk
import { ScrollingModule } from '@angular/cdk/scrolling';

@Component({
  selector: 'app-long-list',
  template: `
    <cdk-virtual-scroll-viewport itemSize="50" class="viewport">
      <div *cdkVirtualFor="let item of items; trackBy: trackByFn" class="item">
        {{ item.name }}
      </div>
    </cdk-virtual-scroll-viewport>
  `,
  styles: [
    `
      .viewport {
        height: 400px;
        width: 100%;
      }
      .item {
        height: 50px;
      }
    `,
  ],
})
export class LongListComponent {
  items = Array.from({ length: 1000 }, (_, i) => ({ id: i, name: `Item ${i}` }));

  trackByFn(index: number, item: any): number {
    return item.id;
  }
}
```

**4. Memoization and Caching:**

```typescript
export class ExpensiveCalculationService {
  private cache = new Map<string, any>();

  calculateExpensiveOperation(key: string, data: any[]): Observable<any> {
    if (this.cache.has(key)) {
      return of(this.cache.get(key));
    }

    return this.performCalculation(data).pipe(tap((result) => this.cache.set(key, result)));
  }

  private performCalculation(data: any[]): Observable<any> {
    // Expensive calculation logic
    return of(/* calculated result */);
  }
}

// Memoization with RxJS
@Injectable()
export class MemoizedService {
  private calculationCache = new Map<string, Observable<any>>();

  getExpensiveData(params: any): Observable<any> {
    const key = JSON.stringify(params);

    if (!this.calculationCache.has(key)) {
      this.calculationCache.set(
        key,
        this.performExpensiveOperation(params).pipe(
          shareReplay(1), // Cache the result
          catchError((error) => {
            this.calculationCache.delete(key);
            return throwError(error);
          }),
        ),
      );
    }

    return this.calculationCache.get(key)!;
  }
}
```

**5. Bundle Size Optimization:**

```typescript
// webpack.config.js or angular.json optimization
{
  "optimization": {
    "scripts": true,
    "styles": true,
    "fonts": true
  },
  "buildOptimizer": true,
  "vendorChunk": false,
  "commonChunk": false
}

// Tree shaking - only import what you need
import { debounceTime, distinctUntilChanged, switchMap } from 'rxjs/operators';
import { HttpClient } from '@angular/common/http';

// Avoid importing entire libraries
// ❌ import * as _ from 'lodash';
// ✅ import { debounce } from 'lodash-es';
```

**6. Image Optimization:**

```typescript
// Lazy loading images
@Component({
  selector: 'app-image-gallery',
  template: `
    <img
      [src]="image.src"
      [alt]="image.alt"
      loading="lazy"
      (load)="onImageLoad()"
      (error)="onImageError()"
    >
  `
})
export class ImageGalleryComponent {
  onImageLoad() {
    console.log('Image loaded successfully');
  }

  onImageError() {
    console.log('Image failed to load');
  }
}

// Responsive images
<img
  srcset="
    image-small.jpg 480w,
    image-medium.jpg 800w,
    image-large.jpg 1200w
  "
  sizes="(max-width: 480px) 480px,
         (max-width: 800px) 800px,
         1200px"
  src="image-fallback.jpg"
  alt="Responsive image"
>
```

**7. Service Worker and Caching:**

```typescript
// ngsw-config.json
{
  "appData": {
    "appVersion": "1.0.0"
  },
  "assetGroups": [
    {
      "name": "app",
      "installMode": "prefetch",
      "resources": {
        "files": [
          "/favicon.ico",
          "/index.html",
          "/manifest.webmanifest",
          "/*.css",
          "/*.js"
        ]
      }
    },
    {
      "name": "assets",
      "installMode": "lazy",
      "updateMode": "prefetch",
      "resources": {
        "files": [
          "/assets/**"
        ]
      }
    }
  ],
  "dataGroups": [
    {
      "name": "api-performance",
      "urls": [
        "/api/**"
      ],
      "cacheConfig": {
        "strategy": "performance",
        "maxSize": 100,
        "maxAge": "3m"
      }
    }
  ]
}
```

**8. Component Optimization:**

```typescript
// Component with optimized change detection
@Component({
  selector: 'app-optimized-component',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div>
      <h2>{{ title$ | async }}</h2>
      <ul>
        <li *ngFor="let item of items$ | async; trackBy: trackByFn">
          {{ item.name }}
        </li>
      </ul>
    </div>
  `,
})
export class OptimizedComponent implements OnInit, OnDestroy {
  title$ = new BehaviorSubject<string>('');
  items$ = new BehaviorSubject<Item[]>([]);
  private destroy$ = new Subject<void>();

  ngOnInit() {
    // Use async pipe to automatically unsubscribe
    this.loadTitle();
    this.loadItems();
  }

  private loadTitle() {
    this.title$.next('Loading...');
    this.dataService
      .getTitle()
      .pipe(takeUntil(this.destroy$))
      .subscribe((title) => this.title$.next(title));
  }

  private loadItems() {
    this.items$.next([]);
    this.dataService
      .getItems()
      .pipe(takeUntil(this.destroy$))
      .subscribe((items) => this.items$.next(items));
  }

  trackByFn(index: number, item: Item): number {
    return item.id;
  }

  ngOnDestroy() {
    this.destroy$.next();
    this.destroy$.complete();
  }
}
```

**9. RxJS Optimization:**

```typescript
export class OptimizedRxjsComponent implements OnInit, OnDestroy {
  searchResults$ = new BehaviorSubject<any[]>([]);
  private searchSubject = new Subject<string>();
  private destroy$ = new Subject<void>();

  ngOnInit() {
    this.searchSubject
      .pipe(
        debounceTime(300), // Wait 300ms after last input
        distinctUntilChanged(), // Only emit if value changed
        switchMap(
          (
            searchTerm, // Cancel previous requests
          ) =>
            this.searchService.search(searchTerm).pipe(
              catchError((error) => of([])), // Handle errors gracefully
            ),
        ),
        takeUntil(this.destroy$),
        shareReplay(1), // Share results with multiple subscribers
      )
      .subscribe((results) => {
        this.searchResults$.next(results);
      });
  }

  onSearch(searchTerm: string) {
    this.searchSubject.next(searchTerm);
  }

  ngOnDestroy() {
    this.destroy$.next();
    this.destroy$.complete();
  }
}
```

**10. Memory Management:**

```typescript
export class MemoryOptimizedComponent implements OnInit, OnDestroy {
  private subscriptions: Subscription[] = [];
  private intervalId: any;
  private observer: IntersectionObserver;

  ngOnInit() {
    // Store subscriptions to unsubscribe later
    const subscription1 = this.service.getData().subscribe((data) => {
      // Handle data
    });

    const subscription2 = this.service.getUpdates().subscribe((update) => {
      // Handle updates
    });

    this.subscriptions.push(subscription1, subscription2);

    // Set up interval
    this.intervalId = setInterval(() => {
      this.updateData();
    }, 1000);

    // Set up intersection observer
    this.observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          this.loadLazyContent();
        }
      });
    });
  }

  ngOnDestroy() {
    // Clean up subscriptions
    this.subscriptions.forEach((sub) => sub.unsubscribe());

    // Clean up interval
    if (this.intervalId) {
      clearInterval(this.intervalId);
    }

    // Clean up observer
    if (this.observer) {
      this.observer.disconnect();
    }
  }
}
```

**11. Template Optimization:**

```html
<!-- Use trackBy with ngFor -->
<div *ngFor="let item of items; trackBy: trackByFn">{{ item.name }}</div>

<!-- Use ngIf for conditional rendering instead of hidden -->
<div *ngIf="showContent">Content here</div>

<!-- Use ngSwitch for multiple conditions -->
<div [ngSwitch]="userType">
  <div *ngSwitchCase="'admin'">Admin content</div>
  <div *ngSwitchCase="'user'">User content</div>
  <div *ngSwitchDefault>Default content</div>
</div>

<!-- Use pure pipes -->
<p>{{ expensiveData | expensiveCalculation | async }}</p>
```

**12. HTTP Request Optimization:**

```typescript
@Injectable()
export class OptimizedHttpService {
  private cache = new Map<string, Observable<any>>();
  private requestQueue = new Map<string, Observable<any>>();

  getData(url: string): Observable<any> {
    // Check cache first
    if (this.cache.has(url)) {
      return this.cache.get(url)!;
    }

    // Check if request is already in progress
    if (this.requestQueue.has(url)) {
      return this.requestQueue.get(url)!;
    }

    // Make new request
    const request = this.http.get(url).pipe(
      shareReplay(1),
      finalize(() => this.requestQueue.delete(url)),
    );

    this.requestQueue.set(url, request);
    this.cache.set(url, request);

    return request;
  }

  // Cache invalidation
  invalidateCache(url: string) {
    this.cache.delete(url);
  }

  // Batch requests
  batchRequests(urls: string[]): Observable<any[]> {
    const requests = urls.map((url) => this.getData(url));
    return forkJoin(requests);
  }
}
```

**Performance Monitoring:**

```typescript
@Injectable()
export class PerformanceMonitorService {
  private navigationStart = performance.now();

  constructor() {
    this.measurePerformance();
  }

  private measurePerformance() {
    // Measure First Contentful Paint
    new PerformanceObserver((entryList) => {
      for (const entry of entryList.getEntries()) {
        console.log('FCP:', entry.startTime);
      }
    }).observe({ entryTypes: ['paint'] });

    // Measure Largest Contentful Paint
    new PerformanceObserver((entryList) => {
      for (const entry of entryList.getEntries()) {
        console.log('LCP:', entry.startTime);
      }
    }).observe({ entryTypes: ['largest-contentful-paint'] });

    // Measure Cumulative Layout Shift
    new PerformanceObserver((entryList) => {
      let clsValue = 0;
      for (const entry of entryList.getEntries()) {
        if (!(entry as any).hadRecentInput) {
          clsValue += (entry as any).value;
        }
      }
      console.log('CLS:', clsValue);
    }).observe({ entryTypes: ['layout-shift'] });
  }

  measureAngularPerformance() {
    // Measure component initialization time
    const initStart = performance.now();

    // Component initialization code

    const initEnd = performance.now();
    console.log(`Component init time: ${initEnd - initStart}ms`);
  }
}
```

**Best Practices Summary:**

1. **Use OnPush change detection** for better performance
2. **Implement lazy loading** for modules and components
3. **Use virtualization** for long lists
4. **Optimize bundle size** with tree shaking
5. **Implement proper caching** strategies
6. **Use RxJS operators** for efficient data handling
7. **Monitor performance** with built-in tools
8. **Clean up resources** in ngOnDestroy
9. **Optimize images** and assets
10. **Use service workers** for caching
11. **Implement memoization** for expensive calculations
12. **Use trackBy** with ngFor for better rendering performance

### Q30: How do you implement lazy loading in Angular applications?

**Answer:**
Lazy loading is a technique that allows Angular applications to load feature modules only when they are needed, significantly improving initial load times and performance.

**1. Basic Lazy Loading Setup:**

```typescript
// app-routing.module.ts
const routes: Routes = [
  { path: '', component: HomeComponent },
  {
    path: 'users',
    loadChildren: () => import('./users/users.module').then((m) => m.UsersModule),
  },
  {
    path: 'products',
    loadChildren: () => import('./products/products.module').then((m) => m.ProductsModule),
  },
  {
    path: 'admin',
    loadChildren: () => import('./admin/admin.module').then((m) => m.AdminModule),
    canLoad: [AdminGuard], // Prevent loading if not authorized
  },
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule],
})
export class AppRoutingModule {}
```

**2. Feature Module Structure:**

```typescript
// users.module.ts
@NgModule({
  declarations: [UserListComponent, UserDetailComponent, UserFormComponent],
  imports: [CommonModule, UsersRoutingModule, ReactiveFormsModule, SharedModule],
  providers: [UserService, UserResolver],
})
export class UsersModule {}

// users-routing.module.ts
const routes: Routes = [
  { path: '', component: UserListComponent },
  { path: ':id', component: UserDetailComponent },
  { path: ':id/edit', component: UserFormComponent },
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule],
})
export class UsersRoutingModule {}
```

**3. Preloading Strategies:**

```typescript
// Custom preloading strategy
@Injectable()
export class CustomPreloadingStrategy implements PreloadingStrategy {
  preload(route: Route, load: Function): Observable<any> {
    // Preload based on custom logic
    if (route.data && route.data['preload']) {
      console.log(`Preloading: ${route.path}`);
      return load();
    }
    return of(null);
  }
}

// Usage in app-routing.module.ts
@NgModule({
  imports: [
    RouterModule.forRoot(routes, {
      preloadingStrategy: CustomPreloadingStrategy,
    }),
  ],
})
export class AppRoutingModule {}

// Route configuration with preloading
const routes: Routes = [
  {
    path: 'dashboard',
    loadChildren: () => import('./dashboard/dashboard.module').then((m) => m.DashboardModule),
    data: { preload: true },
  },
  {
    path: 'settings',
    loadChildren: () => import('./settings/settings.module').then((m) => m.SettingsModule),
    data: { preload: false },
  },
];
```

**4. Conditional Lazy Loading:**

```typescript
// Role-based lazy loading
@Injectable()
export class RoleBasedPreloadingStrategy implements PreloadingStrategy {
  constructor(private authService: AuthService) {}

  preload(route: Route, load: Function): Observable<any> {
    if (route.data && route.data['roles']) {
      const userRoles = this.authService.getUserRoles();
      const requiredRoles = route.data['roles'] as string[];

      const hasRequiredRole = requiredRoles.some((role) => userRoles.includes(role));

      if (hasRequiredRole) {
        return load();
      }
    }

    return of(null);
  }
}

// Route configuration
const routes: Routes = [
  {
    path: 'admin',
    loadChildren: () => import('./admin/admin.module').then((m) => m.AdminModule),
    data: { roles: ['admin'] },
  },
  {
    path: 'manager',
    loadChildren: () => import('./manager/manager.module').then((m) => m.ManagerModule),
    data: { roles: ['admin', 'manager'] },
  },
];
```

**5. Dynamic Module Loading:**

```typescript
// Dynamic module loader service
@Injectable()
export class DynamicModuleLoaderService {
  private loadedModules = new Set<string>();

  async loadModule(modulePath: string, moduleName: string): Promise<any> {
    if (this.loadedModules.has(moduleName)) {
      return Promise.resolve();
    }

    try {
      const module = await import(modulePath);
      this.loadedModules.add(moduleName);
      return module[moduleName];
    } catch (error) {
      console.error(`Failed to load module ${moduleName}:`, error);
      throw error;
    }
  }
}

// Usage in component
export class DynamicLoaderComponent {
  constructor(private moduleLoader: DynamicModuleLoaderService) {}

  async loadFeatureModule() {
    try {
      const FeatureModule = await this.moduleLoader.loadModule(
        './features/feature.module',
        'FeatureModule',
      );
      // Use the loaded module
    } catch (error) {
      console.error('Module loading failed:', error);
    }
  }
}
```

**6. Lazy Loading with Guards:**

```typescript
// CanLoad guard for lazy loading
@Injectable()
export class ModuleGuard implements CanLoad {
  constructor(
    private authService: AuthService,
    private router: Router,
  ) {}

  canLoad(route: Route, segments: UrlSegment[]): boolean | Observable<boolean> | Promise<boolean> {
    const module = route.path;

    // Check if user has permission to load this module
    if (this.authService.hasModulePermission(module)) {
      return true;
    }

    // Redirect to unauthorized page
    this.router.navigate(['/unauthorized']);
    return false;
  }
}

// Route configuration
const routes: Routes = [
  {
    path: 'premium',
    loadChildren: () => import('./premium/premium.module').then((m) => m.PremiumModule),
    canLoad: [ModuleGuard],
  },
];
```

**7. Lazy Loading Components:**

```typescript
// Dynamic component loader
@Component({
  selector: 'app-dynamic-component',
  template: `
    <ng-container #dynamicComponentContainer></ng-container>
  `,
})
export class DynamicComponentLoaderComponent implements OnInit {
  @ViewChild('dynamicComponentContainer', { read: ViewContainerRef }) container!: ViewContainerRef;

  constructor(
    private componentFactoryResolver: ComponentFactoryResolver,
    private moduleLoader: DynamicModuleLoaderService,
  ) {}

  async loadComponent(componentPath: string, componentName: string) {
    try {
      const componentModule = await import(componentPath);
      const componentFactory = this.componentFactoryResolver.resolveComponentFactory(
        componentModule[componentName],
      );

      this.container.clear();
      const componentRef = this.container.createComponent(componentFactory);

      // Pass inputs to component
      componentRef.instance.data = this.someData;

      // Listen to outputs
      componentRef.instance.event.subscribe((event) => {
        console.log('Event from dynamic component:', event);
      });
    } catch (error) {
      console.error('Failed to load component:', error);
    }
  }
}
```

**8. Bundle Analysis and Optimization:**

```json
// package.json scripts
{
  "scripts": {
    "build:analyze": "ng build --stats-json && npx webpack-bundle-analyzer dist/stats.json",
    "build:prod": "ng build --configuration production",
    "build:lazy": "ng build --named-chunks"
  }
}

// angular.json optimization
{
  "projects": {
    "your-app": {
      "architect": {
        "build": {
          "options": {
            "optimization": true,
            "buildOptimizer": true,
            "vendorChunk": false,
            "commonChunk": false,
            "namedChunks": true,
            "extractLicenses": true,
            "sourceMap": false,
            "namedChunks": true
          }
        }
      }
    }
  }
}
```

**9. Performance Monitoring for Lazy Loading:**

```typescript
@Injectable()
export class LazyLoadingMonitorService {
  private loadTimes = new Map<string, number>();

  recordModuleLoad(moduleName: string, loadTime: number) {
    this.loadTimes.set(moduleName, loadTime);
    console.log(`Module ${moduleName} loaded in ${loadTime}ms`);
  }

  getLoadTimes(): Map<string, number> {
    return this.loadTimes;
  }

  getAverageLoadTime(): number {
    const times = Array.from(this.loadTimes.values());
    return times.length > 0 ? times.reduce((a, b) => a + b) / times.length : 0;
  }
}

// Integration with preloading strategy
@Injectable()
export class MonitoredPreloadingStrategy implements PreloadingStrategy {
  constructor(private monitor: LazyLoadingMonitorService) {}

  preload(route: Route, load: Function): Observable<any> {
    const startTime = performance.now();

    return load().pipe(
      tap(() => {
        const loadTime = performance.now() - startTime;
        this.monitor.recordModuleLoad(route.path!, loadTime);
      }),
      catchError((error) => {
        console.error(`Failed to preload ${route.path}:`, error);
        return of(null);
      }),
    );
  }
}
```

**10. Code Splitting Strategies:**

```typescript
// Feature-based code splitting
const routes: Routes = [
  // Core features (eager loaded)
  { path: 'home', component: HomeComponent },
  { path: 'about', component: AboutComponent },

  // Feature modules (lazy loaded)
  {
    path: 'dashboard',
    loadChildren: () => import('./dashboard/dashboard.module').then((m) => m.DashboardModule),
    data: { preload: true }, // Preload important features
  },
  {
    path: 'analytics',
    loadChildren: () => import('./analytics/analytics.module').then((m) => m.AnalyticsModule),
    data: { preload: false }, // Don't preload
  },

  // Third-party integrations (lazy loaded)
  {
    path: 'reports',
    loadChildren: () => import('./reports/reports.module').then((m) => m.ReportsModule),
  },
];
```

**11. Lazy Loading Best Practices:**

```typescript
// 1. Use meaningful chunk names
{
  path: 'users',
  loadChildren: () => import(/* webpackChunkName: "users-module" */ './users/users.module').then(m => m.UsersModule)
}

// 2. Implement proper error handling
@Injectable()
export class ErrorHandlingPreloadingStrategy implements PreloadingStrategy {
  preload(route: Route, load: Function): Observable<any> {
    return load().pipe(
      catchError(error => {
        console.error(`Failed to preload ${route.path}:`, error);
        // Optionally redirect to error page or show notification
        return of(null);
      })
    );
  }
}

// 3. Use intersection observer for predictive loading
@Injectable()
export class PredictivePreloadingStrategy implements PreloadingStrategy {
  private observer: IntersectionObserver;

  constructor() {
    this.observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          // Trigger preloading when route link is visible
          this.preloadRoute(entry.target.getAttribute('data-route'));
        }
      });
    });
  }

  preload(route: Route, load: Function): Observable<any> {
    // Observe route links in DOM
    const links = document.querySelectorAll(`a[href="/${route.path}"]`);
    links.forEach(link => this.observer.observe(link));

    return of(null); // Don't preload immediately
  }

  private preloadRoute(routePath: string) {
    // Find route and trigger loading
    // Implementation depends on your routing setup
  }
}
```

**Performance Benefits of Lazy Loading:**

1. **Reduced Initial Bundle Size**: Only load necessary code initially
2. **Faster Initial Load**: Users can start using the app faster
3. **Better User Experience**: Perceived performance improvement
4. **Bandwidth Optimization**: Users only download what they need
5. **Memory Efficiency**: Unused modules don't consume memory

**Monitoring and Debugging:**

```typescript
// Development tools for monitoring lazy loading
export class LazyLoadingDebugService {
  constructor() {
    // Monitor network requests for chunk loading
    if (typeof window !== 'undefined') {
      const originalImport = window.__webpack_require__;
      window.__webpack_require__ = function (...args) {
        console.log('Loading chunk:', args);
        return originalImport.apply(this, args);
      };
    }
  }
}
```

**Best Practices Summary:**

1. **Identify logical boundaries** for feature modules
2. **Use appropriate preloading strategies** based on user behavior
3. **Implement proper error handling** for failed loads
4. **Monitor performance** and adjust strategies accordingly
5. **Use meaningful chunk names** for better debugging
6. **Consider user experience** when deciding what to preload
7. **Test thoroughly** in different network conditions
8. **Use bundle analysis tools** to optimize chunk sizes
