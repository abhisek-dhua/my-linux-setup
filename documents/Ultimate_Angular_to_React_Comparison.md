# Angular to React Comparison Guide

Learn React by leveraging your Angular knowledge. This guide maps Angular concepts to React equivalents.

## Table of Contents
- [Architecture & Core Concepts](#architecture--core-concepts)
- [Components](#components)
- [Templates & JSX](#templates--jsx)
- [Data Binding](#data-binding)
- [Directives vs React Patterns](#directives-vs-react-patterns)
- [Services & Dependency Injection](#services--dependency-injection)
- [Routing](#routing)
- [Forms](#forms)
- [State Management](#state-management)
- [Lifecycle Hooks](#lifecycle-hooks)
- [HTTP & API Calls](#http--api-calls)
- [Testing](#testing)
- [Build & Development](#build--development)
- [Advanced React Patterns](#advanced-react-patterns)
- [Performance Optimization](#performance-optimization)
- [Advanced TypeScript Patterns](#advanced-typescript-patterns)
- [Error Handling & Debugging](#error-handling--debugging)
- [Security Best Practices](#security-best-practices)
- [Migration Guide](#migration-guide)
- [Modern React Ecosystem](#modern-react-ecosystem)
- [Common Pitfalls & Best Practices](#common-pitfalls--best-practices)
- [Real-World Examples](#real-world-examples)

## Architecture & Core Concepts

### Angular
- Framework-based with comprehensive tooling
- TypeScript-first with strong typing
- Modular architecture with NgModules
- Dependency injection system
- RxJS for reactive programming

### React
- Library-focused (UI only)
- JavaScript with optional TypeScript
- Component-based architecture
- No built-in DI (use Context or external libraries)
- No built-in reactive patterns (useState, useEffect)

**Key Difference**: Angular is a complete framework, React is a UI library.

## Components

### Angular Component (Modern Angular 17+)
```typescript
import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-user',
  standalone: true,
  imports: [CommonModule],
  template: '<div>{{ user().name }}</div>',
  styles: [`
    div {
      padding: 1rem;
      border: 1px solid #ccc;
      border-radius: 8px;
    }
  `]
})
export class UserComponent {
  // Modern signal-based input
  user = input.required<User>();
  
  // Modern output function
  updateUser = output<User>();
  
  handleUpdate() {
    this.updateUser.emit(this.user());
  }
}

// Alternative with traditional decorators (still supported)
@Component({
  selector: 'app-user',
  standalone: true,
  imports: [CommonModule],
  template: '<div>{{ user.name }}</div>',
  styleUrls: ['./user.component.css']
})
export class UserComponent {
  @Input() user: User;
  @Output() updateUser = new EventEmitter<User>();
}
```

### React Component (Modern React 18+)
```tsx
import { useState, useTransition, useDeferredValue } from 'react';

interface UserProps {
  user: User;
  onUpdateUser: (user: User) => void;
}

interface User {
  id: string;
  name: string;
}

function User({ user, onUpdateUser }: UserProps) {
  const [isPending, startTransition] = useTransition();
  const deferredUser = useDeferredValue(user);
  
  const handleUpdate = (updatedUser: User) => {
    startTransition(() => {
      onUpdateUser(updatedUser);
    });
  };
  
  return (
    <div>
      <h3>{deferredUser.name}</h3>
      {isPending && <small>Saving...</small>}
    </div>
  );
}

export default User;
```

**Key Differences**:
- Angular: Standalone components with signals, decorators still supported
- React: Function components with hooks, concurrent features
- Angular: Input signals (`user()`) vs traditional `@Input()`
- React: Props as typed parameters, concurrent updates with `useTransition`
- Angular: Component outputs with signals or EventEmitter
- React: Callback props with deferred updates for performance

## Templates & JSX

### Angular Template (Modern Control Flow)
```html
@if (showContent) {
  <div>
    <h1>{{ title }}</h1>
    <ul>
      @for (item of items; track item.id) {
        <li>{{ item.name }}</li>
      } @empty {
        <li>No items available</li>
      }
    </ul>
  </div>
}

<!-- Legacy syntax (still supported) -->
<div *ngIf="showContent">
  <h1>{{ title }}</h1>
  <ul>
    <li *ngFor="let item of items; trackBy: trackByFn">{{ item.name }}</li>
  </ul>
</div>
```

### React TSX
```tsx
type Item = {
  id: string;
  name: string;
};

{showContent && (
  <div>
    <h1>{title}</h1>
    <ul>
      {items.length > 0 ? (
        items.map((item: Item) => (
          <li key={item.id}>{item.name}</li>
        ))
      ) : (
        <li>No items available</li>
      )}
    </ul>
  </div>
)}
```

### Angular Control Flow Deep Dive

**Modern Angular 17+ Control Flow:**
```html
@if (user) {
  <h2>Welcome, {{ user.name }}!</h2>
} @else if (isLoading) {
  <p>Loading user data...</p>
} @else {
  <p>Please log in</p>
}

@for (item of items; track item.id; let i = $index) {
  <div>
    {{ i + 1 }}. {{ item.name }}
  </div>
} @empty {
  <p>No items found</p>
}

@switch (user.role) {
  @case ('admin') {
    <admin-panel />
  }
  @case ('user') {
    <user-panel />
  }
  @default {
    <guest-panel />
  }
}
```

**React Conditional Patterns:**
```tsx
// @if/@else logic
{user ? (
  <h2>Welcome, {user.name}!</h2>
) : isLoading ? (
  <p>Loading user data...</p>
) : (
  <p>Please log in</p>
)}

// @for/@empty logic
{items.length > 0 ? (
  items.map((item, i) => (
    <div key={item.id}>
      {i + 1}. {item.name}
    </div>
  ))
) : (
  <p>No items found</p>
)}

// @switch logic
{(() => {
  switch (user?.role) {
    case 'admin':
      return <AdminPanel />;
    case 'user':
      return <UserPanel />;
    default:
      return <GuestPanel />;
  }
})()}

// More readable switch with component
function UserRolePanel({ user }: { user: User | null }) {
  switch (user?.role) {
    case 'admin':
      return <AdminPanel />;
    case 'user':
      return <UserPanel />;
    default:
      return <GuestPanel />;
  }
}
```

**Key Differences**:
- Angular `@if/@for` vs React conditional expressions and `.map()`
- Angular `track item.id` vs React `key={item.id}`
- Angular `@empty` vs React ternary operator or early return
- Angular `@switch` vs React `{}` expressions with switch logic or separate components
- Angular `$index` vs React index parameter in `.map()`
- Expressions wrapped in `{}` instead of `{{}}`
- Strong typing with types and generics
- Type safety for props and state (inferred when possible)

## Data Binding

### Angular Data Binding
```html
<!-- One-way interpolation -->
<h1>{{ title }}</h1>

<!-- Property binding -->
<input [value]="name">

<!-- Event binding -->
<button (click)="handleSubmit()">

<!-- Two-way binding -->
<input [(ngModel)]="name">
```

### React Data Binding
```tsx
// One-way rendering
<h1>{title}</h1>

// Property binding
<input value={name} />

// Event binding
<button onClick={handleSubmit}>

// Two-way binding (manual)
<input 
  value={name}
  onChange={(e) => setName(e.target.value)}
/>
```

**Key Differences**:
- No built-in two-way binding in React
- Event handlers use camelCase (types inferred)
- All data flows one-way by default
- Optional typing for event handlers when needed

## Directives vs React Patterns

### Structural Directives (Modern Angular Control Flow)
```typescript
// Modern Angular @if (v17+)
@if (isLoading) {
  <div>Loading...</div>
} @else {
  <div>Content loaded</div>
}

// Angular *ngIf (legacy, still supported)
<div *ngIf="isLoading; else noContent">Loading...</div>
<ng-template #noContent>No content available</ng-template>

// React conditional rendering
{isLoading ? (
  <div>Loading...</div>
) : (
  <div>Content loaded</div>
)}
```

### Attribute Directives (Modern Angular Bindings)
```typescript
// Modern Angular class/style bindings
<div 
  [class.active]="isActive" 
  [style.color]="isActive ? 'green' : 'red'"
  [class.pending]="isPending"
>

// React conditional classes/styles
<div 
  className={`${isActive ? 'active' : ''} ${isPending ? 'pending' : ''}`.trim()}
  style={{color: isActive ? 'green' : 'red'}}
>
```

### Custom Directives (Modern Angular vs React Patterns)
```typescript
// Modern Angular custom directive
@Directive({
  selector: '[appHighlight]',
  standalone: true
})
export class HighlightDirective {
  isHighlighted = signal(false);
  
  @HostListener('mouseenter') 
  onMouseEnter() {
    this.isHighlighted.set(true);
  }
  
  @HostListener('mouseleave') 
  onMouseLeave() {
    this.isHighlighted.set(false);
  }
  
  @HostBinding('class.highlighted')
  get highlighted() {
    return this.isHighlighted();
  }
}

// React - use custom hook for behavior
function useHighlight() {
  const [isHighlighted, setIsHighlighted] = useState(false);
  
  const handlers = useMemo(() => ({
    onMouseEnter: () => setIsHighlighted(true),
    onMouseLeave: () => setIsHighlighted(false)
  }), []);
  
  return [isHighlighted, handlers] as const;
}

// Usage in React
function HighlightedComponent() {
  const [isHighlighted, handlers] = useHighlight();
  
  return (
    <div 
      className={isHighlighted ? 'highlighted' : ''}
      {...handlers}
    >
      Hover me!
    </div>
  );
}
```

## Services & Dependency Injection

### Angular Service
```typescript
@Injectable({
  providedIn: 'root'
})
export class UserService {
  constructor(private http: HttpClient) {}
  
  getUsers(): Observable<User[]> {
    return this.http.get<User[]>('/api/users');
  }
}

// In component
constructor(private userService: UserService) {}
```

### React Service Pattern
```typescript
// userService.ts
import axios from 'axios';

export type User = {
  id: string;
  name: string;
  email: string;
};

export class UserService {
  private api = axios.create({ baseURL: '/api' });
  
  async getUsers(): Promise<User[]> {
    const response = await this.api.get<User[]>('/users');
    return response.data;
  }
}

// Using in component
function UserList() {
  const [users, setUsers] = useState<User[]>([]);
  const userService = useMemo(() => new UserService(), []);
  
  useEffect(() => {
    userService.getUsers().then(setUsers);
  }, [userService]);
  
  return (
    <div>
      {users.map(user => (
        <div key={user.id}>{user.name}</div>
      ))}
    </div>
  );
}
```

**React Alternatives to DI**:
- Context API for global services
- Custom hooks for service instances
- External libraries (React DI, Inversify)

## Routing

### Angular Routing (Modern Angular 17+ with Standalone Components)
```typescript
import { Routes, provideRouter, withComponentInputBinding } from '@angular/router';
import { bootstrapApplication } from '@angular/platform-browser';
import { UserComponent } from './user/user.component';
import { AboutComponent } from './about/about.component';

export const routes: Routes = [
  {
    path: 'users/:id',
    component: UserComponent,
    resolve: {
      user: (route: ActivatedRouteSnapshot) => {
        return inject(UserService).getUser(route.paramMap.get('id')!);
      }
    }
  },
  { 
    path: 'about', 
    component: AboutComponent 
  },
  { 
    path: '', 
    redirectTo: '/users', 
    pathMatch: 'full' 
  }
];

// Bootstrap with standalone components
bootstrapApplication(AppComponent, {
  providers: [
    provideRouter(routes, withComponentInputBinding())
  ]
});

// In component template
@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, RouterOutlet, RouterLink],
  template: `
    <nav>
      <a [routerLink]="['/users', userId]">User</a>
      <a routerLink="/about">About</a>
    </nav>
    <router-outlet></router-outlet>
  `
})
export class AppComponent {
  userId = '123';
}
```

### React Router (Modern React Router v6.4+ with Data Router)
```tsx
import { 
  createBrowserRouter, 
  RouterProvider, 
  Link, 
  useParams, 
  useLoaderData,
  Outlet
} from 'react-router-dom';
import { z } from 'zod';

// Type-safe route parameters
const userParamsSchema = z.object({
  id: z.string()
});

function UserComponent() {
  const params = useParams();
  const { id } = userParamsSchema.parse(params);
  const user = useLoaderData() as User;
  
  return <div>User: {user.name} (ID: {id})</div>;
}

function AboutComponent() {
  return <div>About Page</div>;
}

function RootLayout() {
  return (
    <div>
      <nav>
        <Link to="/users/123">User</Link>
        <Link to="/about">About</Link>
      </nav>
      <Outlet />
    </div>
  );
}

// Modern data router with loaders and actions
const router = createBrowserRouter([
  {
    path: "/",
    element: <RootLayout />,
    errorElement: <div>Something went wrong!</div>,
    children: [
      {
        path: "users/:id",
        element: <UserComponent />,
        loader: async ({ params }) => {
          const { id } = userParamsSchema.parse(params);
          const response = await fetch(`/api/users/${id}`);
          if (!response.ok) {
            throw new Response("User not found", { status: 404 });
          }
          return response.json();
        },
        action: async ({ request, params }) => {
          const { id } = userParamsSchema.parse(params);
          const formData = await request.formData();
          const userData = Object.fromEntries(formData);
          
          const response = await fetch(`/api/users/${id}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(userData),
          });
          
          if (!response.ok) {
            throw new Response("Update failed", { status: 400 });
          }
          
          return response.json();
        },
      },
      { 
        path: "about", 
        element: <AboutComponent /> 
      },
    ],
  },
]);

function App() {
  return <RouterProvider router={router} />;
}
```

**Key Differences**:
- No router-outlet - use Routes/Route components
- Link component instead of routerLink
- Route parameters via useParams hook

## Forms

### Angular Forms (Modern Angular 17+)
```typescript
import { Component, signal, computed, inject, OnInit } from '@angular/core';
import { 
  FormBuilder, 
  ReactiveFormsModule, 
  Validators, 
  FormGroup,
  FormControl 
} from '@angular/forms';

@Component({
  selector: 'app-user-form',
  standalone: true,
  imports: [ReactiveFormsModule, CommonModule],
  template: `
    <form [formGroup]="form" (ngSubmit)="onSubmit()">
      <input formControlName="name" placeholder="Name">
      @if (form.invalid) {
        <small>Name is required</small>
      }
      <button type="submit" [disabled]="form.invalid || isSubmitting()">
        Submit
      </button>
    </form>
  `
})
export class UserFormComponent implements OnInit {
  private fb = inject(FormBuilder);
  
  // Modern nonNullable form builder
  form = this.fb.nonNullable.group({
    name: ['', Validators.required]
  });
  
  // Signal-based state
  userSignal = signal<User | null>(null);
  isSubmitting = signal(false);
  
  // Computed signal for form validation
  canSubmit = computed(() => 
    this.form.valid && !this.isSubmitting()
  );
  
  ngOnInit() {
    // React to form changes with signals
    this.form.valueChanges.subscribe(value => {
      if (this.form.valid) {
        this.userSignal.set(value as User);
      }
    });
  }
  
  async onSubmit() {
    if (this.form.valid) {
      this.isSubmitting.set(true);
      try {
        const userData = this.form.getRawValue();
        // Submit logic here
        console.log('Form submitted:', userData);
      } finally {
        this.isSubmitting.set(false);
      }
    }
  }
}

// Legacy template-driven (still supported)
@Component({
  selector: 'app-template-form',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <input [(ngModel)]="user.name" name="name" required>
    @if (!user.name) {
      <small>Name is required</small>
    }
  `
})
export class TemplateFormComponent {
  user = signal<User>({ id: '', name: '' });
}
```

### React Forms (Modern React 18+ with TypeScript)
```tsx
import { useState, useTransition, FormEvent } from 'react';
import { useForm, SubmitHandler } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';

// Modern form with React Hook Form + Zod validation
const userSchema = z.object({
  name: z.string().min(1, 'Name is required'),
  email: z.string().email('Valid email required')
});

type UserFormData = z.infer<typeof userSchema>;

function ModernUserForm() {
  const [isPending, startTransition] = useTransition();
  
  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting }
  } = useForm<UserFormData>({
    resolver: zodResolver(userSchema)
  });
  
  const onSubmit: SubmitHandler<UserFormData> = async (data) => {
    startTransition(async () => {
      try {
        // API call here
        console.log('Form data:', data);
      } catch (error) {
        console.error('Submit error:', error);
      }
    });
  };
  
  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <div>
        <input
          {...register('name')}
          placeholder="Name"
          aria-invalid={errors.name ? 'true' : 'false'}
          disabled={isSubmitting || isPending}
        />
        {errors.name && <span className="error">{errors.name.message}</span>}
      </div>
      
      <div>
        <input
          {...register('email')}
          type="email"
          placeholder="Email"
          aria-invalid={errors.email ? 'true' : 'false'}
          disabled={isSubmitting || isPending}
        />
        {errors.email && <span className="error">{errors.email.message}</span>}
      </div>
      
      <button 
        type="submit" 
        disabled={isSubmitting || isPending}
      >
        {isSubmitting || isPending ? 'Submitting...' : 'Submit'}
      </button>
    </form>
  );
}

// Traditional controlled component approach
interface FormData {
  name: string;
  email: string;
}

function ControlledUserForm() {
  const [formData, setFormData] = useState<FormData>({
    name: '',
    email: ''
  });
  
  const [isPending, startTransition] = useTransition();
  
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };
  
  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    
    startTransition(() => {
      // Submit logic here
      console.log('Form data:', formData);
    });
  };
  
  return (
    <form onSubmit={handleSubmit}>
      <input
        name="name"
        value={formData.name}
        onChange={handleChange}
        placeholder="Name"
        disabled={isPending}
      />
      <input
        name="email"
        type="email"
        value={formData.email}
        onChange={handleChange}
        placeholder="Email"
        disabled={isPending}
      />
      <button type="submit" disabled={isPending}>
        {isPending ? 'Submitting...' : 'Submit'}
      </button>
    </form>
  );
}
```

**Popular React Form Libraries**:
- React Hook Form (similar to Angular reactive forms)
- Formik
- Final Form

## State Management

### Angular Services + RxJS
```typescript
@Injectable()
export class StoreService {
  private state$ = new BehaviorSubject<State>(initialState);
  
  setState(newState: Partial<State>) {
    const current = this.state$.value;
    this.state$.next({ ...current, ...newState });
  }
}
```

### React State Options
```tsx
// useState for local state
const [count, setCount] = useState(0);

// useReducer for complex state
type State = {
  count: number;
  loading: boolean;
};

type Action = 
  | { type: 'INCREMENT' }
  | { type: 'DECREMENT' }
  | { type: 'SET_LOADING'; payload: boolean };

const [state, dispatch] = useReducer(reducer, initialState);

// Context for global state
type AppContextType = {
  user: User | null;
  setUser: (user: User | null) => void;
};

const AppContext = createContext<AppContextType | undefined>(undefined);

// External libraries
// Redux Toolkit (most popular)
// Zustand (simpler alternative)
// Jotai (atomic state management)
```

## Lifecycle Hooks

### Angular Lifecycle (Modern Angular 17+ with Signals)
```typescript
import { 
  Component, 
  OnInit, 
  OnChanges, 
  AfterViewInit, 
  OnDestroy, 
  SimpleChanges,
  effect,
  DestroyRef,
  inject
} from '@angular/core';

@Component({
  selector: 'app-user',
  standalone: true,
  imports: [CommonModule],
  template: `<div>{{ user().name }}</div>`
})
export class UserComponent implements OnInit, OnChanges, AfterViewInit, OnDestroy {
  user = input.required<User>();
  
  // Modern effect for reactive changes
  private userEffect = effect(() => {
    console.log('User changed:', this.user().name);
  });
  
  // Traditional lifecycle hooks
  ngOnInit() {
    console.log('Component initialized');
  }
  
  ngOnChanges(changes: SimpleChanges) {
    if (changes['user']) {
      console.log('User input changed');
    }
  }
  
  ngAfterViewInit() {
    console.log('View rendered');
  }
  
  ngOnDestroy() {
    console.log('Component destroyed');
  }
  
  // Modern destroy ref for cleanup
  constructor() {
    const destroyRef = inject(DestroyRef);
    
    // Auto-cleanup on destroy
    destroyRef.onDestroy(() => {
      console.log('Component destroyed via DestroyRef');
    });
  }
}
```

### React Hooks (Modern React 18+ with Concurrent Features)
```tsx
import { 
  useState, 
  useEffect, 
  useTransition, 
  useDeferredValue,
  useLayoutEffect,
  useCallback,
  useMemo
} from 'react';

interface ComponentProps {
  id: string;
}

function Component({ id }: ComponentProps) {
  const [data, setData] = useState<User[] | null>(null);
  const [isPending, startTransition] = useTransition();
  
  // Deferred value for expensive renders
  const deferredData = useDeferredValue(data);
  
  // Memoized callback for API calls
  const fetchData = useCallback(async (id: string) => {
    try {
      const response = await fetch(`/api/data/${id}`);
      const result = await response.json();
      
      // Use startTransition for non-urgent updates
      startTransition(() => {
        setData(result);
      });
    } catch (error) {
      console.error('Failed to fetch data:', error);
    }
  }, []);
  
  // componentDidMount + componentDidUpdate with proper cleanup
  useEffect(() => {
    fetchData(id);
    
    // Cleanup function (componentWillUnmount)
    return () => {
      // Cancel ongoing requests, cleanup timers, etc.
      console.log('Cleaning up component');
    };
  }, [id, fetchData]);
  
  // Layout effect for synchronous DOM operations
  useLayoutEffect(() => {
    // Synchronous DOM measurements or updates
    console.log('Layout effect - DOM updated');
  });
  
  // Memoized expensive computation
  const processedData = useMemo(() => {
    if (!deferredData) return [];
    
    return deferredData.map(item => ({
      ...item,
      displayName: item.name.toUpperCase(),
      computedValue: expensiveCalculation(item.value)
    }));
  }, [deferredData]);
  
  return (
    <div>
      {isPending && <div>Loading...</div>}
      {processedData.map(item => (
        <div key={item.id}>
          {item.displayName}
        </div>
      ))}
    </div>
  );
}
```

**Hook Mapping**:
- `ngOnInit` → `useEffect(() => {}, [])` or constructor with `inject()`
- `ngOnChanges` → `useEffect(() => {}, [prop])` or `effect(() => signal())`
- `ngOnDestroy` → `useEffect(() => { return cleanup; }, [])` or `DestroyRef`
- `ngAfterViewInit` → `useLayoutEffect(() => {}, [])`
- Angular `signal()` → React `useState()` + `useDeferredValue()`
- Angular `effect()` → React `useEffect()` with dependencies
- Angular `computed()` → React `useMemo()`

## HTTP & API Calls

### Angular HttpClient
```typescript
@Injectable()
export class ApiService {
  constructor(private http: HttpClient) {}
  
  getData(): Observable<any> {
    return this.http.get('/api/data');
  }
}
```

### React HTTP Patterns
```tsx
// Using fetch
type Data = {
  id: string;
  name: string;
};

function DataComponent() {
  const [data, setData] = useState<Data | null>(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    fetch('/api/data')
      .then((res) => res.json())
      .then((result) => {
        setData(result);
        setLoading(false);
      })
      .catch((error) => {
        console.error('Error fetching data:', error);
        setLoading(false);
      });
  }, []);
  
  if (loading) return <div>Loading...</div>;
  return <div>{data?.name}</div>;
}

// Using axios
import axios from 'axios';

function DataComponentWithAxios() {
  const [data, setData] = useState<Data | null>(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get<Data>('/api/data');
        setData(response.data);
      } catch (error) {
        console.error('Error fetching data:', error);
      } finally {
        setLoading(false);
      }
    };
    
    fetchData();
  }, []);
  
  if (loading) return <div>Loading...</div>;
  return <div>{data?.name}</div>;
}
```

**Popular React HTTP Libraries**:
- Axios (most popular)
- TanStack Query (formerly React Query) (caching + state management)
- SWR (stale-while-revalidate)
- Fetch API (built-in)

## Testing

### Angular Testing
```typescript
describe('UserComponent', () => {
  let component: UserComponent;
  let fixture: ComponentFixture<UserComponent>;
  
  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [UserComponent]
    });
    fixture = TestBed.createComponent(UserComponent);
    component = fixture.componentInstance;
  });
});
```

### React Testing
```tsx
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi } from 'vitest'; // or jest
import User from './User';
import Button from './Button';

// Mock data
const mockUser = {
  id: '1',
  name: 'John Doe'
} as const;

describe('UserComponent', () => {
  it('renders user name', () => {
    render(<User user={mockUser} onUpdateUser={() => {}} />);
    expect(screen.getByText('John Doe')).toBeInTheDocument();
  });
  
  it('handles click', async () => {
    const handleClick = vi.fn();
    const user = userEvent.setup();
    
    render(<Button onClick={handleClick}>Click</Button>);
    await user.click(screen.getByText('Click'));
    expect(handleClick).toHaveBeenCalledTimes(1);
  });
  
  it('has correct props typing', () => {
    const onUpdateUser = (user) => {
      console.log('User updated:', user);
    };
    
    expect(() => 
      render(<User user={mockUser} onUpdateUser={onUpdateUser} />)
    ).not.toThrow();
  });
});
```

## Build & Development

### Angular CLI Commands (Modern Angular 17+)
```bash
# Create new standalone app
ng new my-app --standalone --style=css --routing=false

# Development
ng serve

# Build with optimization
ng build --configuration production

# Testing
ng test
ng e2e

# Generate standalone component
ng generate component my-component --standalone

# Modern Angular command for building SSR
ng build --ssr

# Check for updates
ng update
```

### React Commands (Modern React 18+ with Vite)
```bash
# Create new app with Vite (recommended)
npm create vite@latest my-app -- --template react-ts

# Create Next.js app
npx create-next-app@latest my-app --typescript --eslint --tailwind --app

# Development
npm run dev

# Build
npm run build

# Test
npm run test

# Type checking
npm run type-check

# Preview build
npm run preview

# Modern package scripts (package.json)
{
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "test": "vitest",
    "test:ui": "vitest --ui",
    "type-check": "tsc --noEmit",
    "lint": "eslint . --ext ts,tsx --report-unused-disable-directives --max-warnings 0",
    "lint:fix": "eslint . --ext ts,tsx --fix"
  }
}
```

## Advanced React Patterns

### Higher-Order Components (HOCs)

**Angular Equivalent**: Directives, Guards, Decorators

```typescript
// Angular Guard Pattern
@Injectable({
  providedIn: 'root'
})
export class AuthGuard implements CanActivate {
  canActivate(): boolean {
    return this.authService.isLoggedIn();
  }
}

// React HOC Pattern
type WithAuthProps = {
  isAuthenticated: boolean;
};

function withAuth<P extends object>(
  Component: React.ComponentType<P & WithAuthProps>
) {
  return function AuthenticatedComponent(props: P) {
    const isAuthenticated = useAuth();
    
    if (!isAuthenticated) {
      return <div>Please log in</div>;
    }
    
    return <Component {...props} isAuthenticated={isAuthenticated} />;
  };
}

// Usage
const ProtectedComponent = withAuth(({ isAuthenticated, ...props }) => {
  return <div>Protected content</div>;
});
```

### Render Props Pattern

**Angular Equivalent**: ng-template and ngTemplateOutlet

```typescript
// Angular Template Pattern
@Component({
  template: `
@if (data$ | async; as data) {
      <ng-template [ngTemplateOutlet]="contentTemplate" 
                   [ngTemplateOutletContext]="{$implicit: data}">
      </ng-template>
    }
  </div>
  `,
})
export class DataProviderComponent {
  @ContentChild('contentTemplate') contentTemplate: TemplateRef<any>;
}

// React Render Props Pattern
type DataFetcherProps<T> = {
  url: string;
  children: (data: T | null, loading: boolean, error: Error | null) => React.ReactNode;
};

function DataFetcher<T>({ url, children }: DataFetcherProps<T>) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  
  useEffect(() => {
    fetch(url)
      .then(res => res.json())
      .then(setData)
      .catch(setError)
      .finally(() => setLoading(false));
  }, [url]);
  
  return <>{children(data, loading, error)}</>;
}

// Usage
function UserProfile() {
  return (
    <DataFetcher<User> url="/api/user/123">
      {(user, loading, error) => {
        if (loading) return <div>Loading...</div>;
        if (error) return <div>Error: {error.message}</div>;
        return <div>Hello, {user?.name}!</div>;
      }}
    </DataFetcher>
  );
}
```

### Compound Components Pattern

**Angular Equivalent**: Multi-component directives, Content projection

```typescript
// Angular Content Projection
@Component({
  selector: 'app-tabs',
  template: `
    <div class="tabs">
      <div class="tab-headers">
        <ng-content select="[tab-header]"></ng-content>
      </div>
      <div class="tab-content">
        <ng-content select="[tab-content]"></ng-content>
      </div>
    </div>
  `
})
export class TabsComponent {}

// React Compound Components
type TabsContextType = {
  activeTab: string;
  setActiveTab: (id: string) => void;
};

const TabsContext = createContext<TabsContextType | undefined>(undefined);

function Tabs({ children }: { children: React.ReactNode }) {
  const [activeTab, setActiveTab] = useState('tab1');
  
  return (
    <TabsContext.Provider value={{ activeTab, setActiveTab }}>
      <div className="tabs">{children}</div>
    </TabsContext.Provider>
  );
}

function TabHeader({ id, children }: { id: string; children: React.ReactNode }) {
  const context = useContext(TabsContext);
  if (!context) throw new Error('TabHeader must be used within Tabs');
  
  const { activeTab, setActiveTab } = context;
  const isActive = activeTab === id;
  
  return (
    <button
      className={`tab-header ${isActive ? 'active' : ''}`}
      onClick={() => setActiveTab(id)}
    >
      {children}
    </button>
  );
}

function TabContent({ id, children }: { id: string; children: React.ReactNode }) {
  const context = useContext(TabsContext);
  if (!context) throw new Error('TabContent must be used within Tabs');
  
  const { activeTab } = context;
  if (activeTab !== id) return null;
  
  return <div className="tab-content">{children}</div>;
}

// Usage
function App() {
  return (
    <Tabs>
      <div className="tab-headers">
        <TabHeader id="tab1">Profile</TabHeader>
        <TabHeader id="tab2">Settings</TabHeader>
      </div>
      <TabContent id="tab1">
        <div>Profile content</div>
      </TabContent>
      <TabContent id="tab2">
        <div>Settings content</div>
      </TabContent>
    </Tabs>
  );
}
```

### Custom Hooks as Angular Services

```typescript
// Angular Service
@Injectable()
export class UserService {
  private userSubject = new BehaviorSubject<User | null>(null);
  user$ = this.userSubject.asObservable();
  
  constructor(private http: HttpClient) {
    this.loadUser();
  }
  
  private loadUser() {
    this.http.get<User>('/api/user').subscribe(user => {
      this.userSubject.next(user);
    });
  }
  
  updateUser(user: Partial<User>) {
    const current = this.userSubject.value;
    if (current) {
      this.userSubject.next({ ...current, ...user });
    }
  }
}

// React Custom Hook
type User = {
  id: string;
  name: string;
  email: string;
};

type UseUserReturn = {
  user: User | null;
  loading: boolean;
  error: Error | null;
  updateUser: (updates: Partial<User>) => void;
};

function useUser(): UseUserReturn {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  
  useEffect(() => {
    const loadUser = async () => {
      try {
        const response = await fetch('/api/user');
        const userData = await response.json();
        setUser(userData);
      } catch (err) {
        setError(err as Error);
      } finally {
        setLoading(false);
      }
    };
    
    loadUser();
  }, []);
  
  const updateUser = useCallback((updates: Partial<User>) => {
    if (user) {
      setUser(prev => prev ? { ...prev, ...updates } : null);
    }
  }, [user]);
  
  return { user, loading, error, updateUser };
}

// Usage
function UserProfile() {
  const { user, loading, error, updateUser } = useUser();
  
  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;
  if (!user) return <div>No user found</div>;
  
  return (
    <div>
      <h1>{user.name}</h1>
      <button onClick={() => updateUser({ name: 'New Name' })}>
        Update Name
      </button>
    </div>
  );
}
```

## Performance Optimization

### Memoization

**Angular Equivalent**: OnPush change detection strategy

```typescript
// Angular OnPush Strategy
@Component({
  selector: 'app-user-card',
  template: `
    <div class="user-card">
      <h3>{{ user.name }}</h3>
      <p>{{ user.email }}</p>
    </div>
  `,
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class UserCardComponent {
  @Input() user: User;
}

// React Memoization
type UserCardProps = {
  user: User;
};

const UserCard = React.memo(({ user }: UserCardProps) => {
  return (
    <div className="user-card">
      <h3>{user.name}</h3>
      <p>{user.email}</p>
    </div>
  );
}, (prevProps, nextProps) => {
  // Custom comparison function
  return prevProps.user.id === nextProps.user.id &&
         prevProps.user.name === nextProps.user.name &&
         prevProps.user.email === nextProps.user.email;
});

// useMemo for expensive calculations
function ExpensiveComponent({ items }: { items: Item[] }) {
  const expensiveValue = useMemo(() => {
    console.log('Calculating expensive value...');
    return items.reduce((sum, item) => sum + item.value, 0);
  }, [items]);
  
  return <div>Total: {expensiveValue}</div>;
}

// useCallback for memoized functions
type ButtonProps = {
  onClick: () => void;
  children: React.ReactNode;
};

const Button = React.memo(({ onClick, children }: ButtonProps) => {
  console.log('Button rendered');
  return <button onClick={onClick}>{children}</button>;
});

function ParentComponent() {
  const [count, setCount] = useState(0);
  
  const handleClick = useCallback(() => {
    console.log('Button clicked');
  }, []); // Empty dependency array means function never changes
  
  return (
    <div>
      <Button onClick={handleClick}>Click me</Button>
      <button onClick={() => setCount(count + 1)}>Count: {count}</button>
    </div>
  );
}
```

### Lazy Loading and Code Splitting

**Angular Equivalent**: Lazy loaded modules

```typescript
// Angular Lazy Module
const routes: Routes = [
  {
    path: 'admin',
    loadChildren: () => import('./admin/admin.module').then(m => m.AdminModule)
  }
];

// React Lazy Loading
import { lazy, Suspense } from 'react';

const AdminDashboard = lazy(() => import('./AdminDashboard'));
const UserSettings = lazy(() => import('./UserSettings'));

function App() {
  return (
    <div>
      <h1>My App</h1>
      <Suspense fallback={<div>Loading...</div>}>
        <Routes>
          <Route path="/admin" element={<AdminDashboard />} />
          <Route path="/settings" element={<UserSettings />} />
        </Routes>
      </Suspense>
    </div>
  );
}

// Dynamic imports with error boundaries
type LazyComponentProps = {
  factory: () => Promise<{ default: React.ComponentType<any> }>;
  fallback?: React.ReactNode;
};

function LazyComponent({ factory, fallback = <div>Loading...</div> }: LazyComponentProps) {
  const [Component, setComponent] = useState<React.ComponentType<any> | null>(null);
  const [error, setError] = useState<Error | null>(null);
  
  useEffect(() => {
    factory()
      .then(module => setComponent(() => module.default))
      .catch(setError);
  }, [factory]);
  
  if (error) return <div>Error loading component: {error.message}</div>;
  if (!Component) return fallback;
  
  return <Component />;
}

// Usage
<LazyComponent 
  factory={() => import('./HeavyComponent')} 
  fallback={<div>Loading heavy component...</div>} 
/>
```

### Virtualization

**Angular Equivalent**: CDK virtual scrolling

```typescript
// Angular CDK Virtual Scroll
<cdk-virtual-scroll-viewport itemSize="50">
  <div *cdkVirtualFor="let item of items">
    {{ item.name }}
  </div>
</cdk-virtual-scroll-viewport>

// React Virtualization (react-window)
import { FixedSizeList as List } from 'react-window';

type VirtualListProps = {
  items: Item[];
};

function VirtualList({ items }: VirtualListProps) {
  const Row = ({ index, style }: { index: number; style: React.CSSProperties }) => (
    <div style={style}>
      {items[index].name}
    </div>
  );
  
  return (
    <List
      height={400}
      itemCount={items.length}
      itemSize={50}
      width="100%"
    >
      {Row}
    </List>
  );
}
```

### Bundle Optimization

```typescript
// Tree shaking
import { debounce } from 'lodash-es/debounce'; // Instead of entire lodash
// or
import debounce from 'lodash.debounce'; // Individual function

// Dynamic imports for large libraries
function AdminPanel() {
  const [ChartComponent, setChartComponent] = useState<React.ComponentType | null>(null);
  
  useEffect(() => {
    // Load chart library only when needed
    import('./ChartComponent').then(module => {
      setChartComponent(() => module.default);
    });
  }, []);
  
  if (!ChartComponent) return <div>Loading chart...</div>;
  
  return <ChartComponent data={chartData} />;
}
```

## Advanced TypeScript Patterns

### Generic Components

```typescript
// Angular Generic Component
@Component({
  selector: 'app-generic-list',
  template: `
    @for (item of items; track item.id) {
      <div>
        {{ item.property }}
      </div>
    } @empty {
      <div>No items available</div>
    }
  `
})
export class GenericListComponent<T> {
  @Input() items: T[] = [];
  @Input() property: keyof T;
}

// React Generic Component
type GenericListProps<T, K extends keyof T> = {
  items: T[];
  property: K;
  renderItem?: (item: T) => React.ReactNode;
};

function GenericList<T, K extends keyof T>({ 
  items, 
  property, 
  renderItem 
}: GenericListProps<T, K>) {
  return (
    <div>
      {items.map((item, index) => (
        <div key={index}>
          {renderItem ? renderItem(item) : String(item[property])}
        </div>
      ))}
    </div>
  );
}

// Usage examples
interface User {
  id: string;
  name: string;
  email: string;
}

interface Product {
  id: string;
  title: string;
  price: number;
}

function App() {
  const users: User[] = [
    { id: '1', name: 'John', email: 'john@example.com' }
  ];
  
  const products: Product[] = [
    { id: '1', title: 'Laptop', price: 999 }
  ];
  
  return (
    <div>
      <GenericList items={users} property="name" />
      <GenericList items={products} property="title" />
      <GenericList 
        items={products} 
        property="price"
        renderItem={(product) => (
          <div>{product.title}: ${product.price}</div>
        )}
      />
    </div>
  );
}
```

### Discriminated Unions

```typescript
// Type-safe API responses
type ApiResponse<T> = 
  | { status: 'loading' }
  | { status: 'success'; data: T }
  | { status: 'error'; error: string };

type UserData = {
  id: string;
  name: string;
};

function DataDisplay({ response }: { response: ApiResponse<UserData> }) {
  switch (response.status) {
    case 'loading':
      return <div>Loading...</div>;
    case 'success':
      return <div>Hello, {response.data.name}!</div>;
    case 'error':
      return <div>Error: {response.error}</div>;
    default:
      // Exhaustive check - TypeScript will error if we miss a case
      const _exhaustiveCheck: never = response;
      return _exhaustiveCheck;
  }
}

// Complex component props
type ButtonVariant = 'primary' | 'secondary' | 'danger';
type ButtonSize = 'small' | 'medium' | 'large';

interface BaseButtonProps {
  children: React.ReactNode;
  disabled?: boolean;
}

interface PrimaryButtonProps extends BaseButtonProps {
  variant: 'primary';
  onClick: () => void;
}

interface SecondaryButtonProps extends BaseButtonProps {
  variant: 'secondary';
  href: string;
}

interface DangerButtonProps extends BaseButtonProps {
  variant: 'danger';
  onConfirm: () => void;
}

type ButtonProps = PrimaryButtonProps | SecondaryButtonProps | DangerButtonProps;

function Button(props: ButtonProps) {
  const baseClasses = 'btn';
  const variantClasses = {
    primary: 'btn-primary',
    secondary: 'btn-secondary',
    danger: 'btn-danger'
  };
  
  const classes = `${baseClasses} ${variantClasses[props.variant]}`;
  
  switch (props.variant) {
    case 'primary':
      return (
        <button 
          className={classes}
          onClick={props.onClick}
          disabled={props.disabled}
        >
          {props.children}
        </button>
      );
    
    case 'secondary':
      return (
        <a 
          className={classes}
          href={props.href}
        >
          {props.children}
        </a>
      );
    
    case 'danger':
      return (
        <button 
          className={classes}
          onClick={props.onConfirm}
          disabled={props.disabled}
        >
          {props.children}
        </button>
      );
  }
}
```

### Advanced Hook Types

```typescript
// Type-safe custom hooks with overloads
function useApi<T>(url: string): ApiResponse<T>;
function useApi<T, P>(url: string, params: P): ApiResponse<T>;
function useApi<T, P>(url: string, params?: P): ApiResponse<T> {
  const [response, setResponse] = useState<ApiResponse<T>>({ status: 'loading' });
  
  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch(url);
        const data = await res.json();
        setResponse({ status: 'success', data });
      } catch (error) {
        setResponse({ status: 'error', error: String(error) });
      }
    };
    
    fetchData();
  }, [url, params]);
  
  return response;
}

// Generic state management
type Action<T, P = void> = {
  type: T;
  payload?: P;
};

type Reducer<S, A extends Action<any>> = (state: S, action: A) => S;

function useTypedReducer<S, A extends Action<any>>(
  reducer: Reducer<S, A>,
  initialState: S
): [S, React.Dispatch<A>] {
  return useReducer(reducer, initialState);
}

// Example usage
type TodoAction = 
  | Action<'ADD_TODO', { text: string }>
  | Action<'TOGGLE_TODO', { id: number }>
  | Action<'DELETE_TODO', { id: number }>;

type Todo = {
  id: number;
  text: string;
  completed: boolean;
};

type TodoState = Todo[];

const todoReducer: Reducer<TodoState, TodoAction> = (state, action) => {
  switch (action.type) {
    case 'ADD_TODO':
      return [...state, {
        id: Date.now(),
        text: action.payload.text,
        completed: false
      }];
    case 'TOGGLE_TODO':
      return state.map(todo =>
        todo.id === action.payload.id
          ? { ...todo, completed: !todo.completed }
          : todo
      );
    case 'DELETE_TODO':
      return state.filter(todo => todo.id !== action.payload.id);
    default:
      return state;
  }
};

function TodoApp() {
  const [todos, dispatch] = useTypedReducer(todoReducer, []);
  
  return (
    <div>
      {todos.map(todo => (
        <div key={todo.id}>
          <input
            type="checkbox"
            checked={todo.completed}
            onChange={() => dispatch({ type: 'TOGGLE_TODO', payload: { id: todo.id } })}
          />
          <span>{todo.text}</span>
          <button onClick={() => dispatch({ type: 'DELETE_TODO', payload: { id: todo.id } })}>
            Delete
          </button>
        </div>
      ))}
    </div>
  );
}
```

### ForwardRef with Types

```typescript
import React, { forwardRef, ForwardedRef } from 'react';

type FancyInputProps = {
  label: string;
  error?: string;
  value: string;
  onChange: (value: string) => void;
};

const FancyInput = forwardRef<HTMLInputElement, FancyInputProps>(
  ({ label, error, value, onChange }, ref: ForwardedRef<HTMLInputElement>) => {
    return (
      <div className="fancy-input">
        <label>{label}</label>
        <input
          ref={ref}
          value={value}
          onChange={e => onChange(e.target.value)}
          className={error ? 'error' : ''}
        />
        {error && <span className="error-message">{error}</span>}
      </div>
    );
  }
);

FancyInput.displayName = 'FancyInput';

// Usage with ref
function Form() {
  const inputRef = useRef<HTMLInputElement>(null);
  
  const focusInput = () => {
    inputRef.current?.focus();
  };
  
  return (
    <div>
      <FancyInput
        ref={inputRef}
        label="Name"
        value=""
        onChange={() => {}}
      />
      <button onClick={focusInput}>Focus Input</button>
    </div>
  );
}
```

## Error Handling & Debugging

### Error Boundaries

**Angular Equivalent**: ErrorHandler service

```typescript
// Angular Global Error Handler
@Injectable()
export class GlobalErrorHandler implements ErrorHandler {
  handleError(error: any): void {
    console.error('Global error:', error);
    // Send to logging service
  }
}

// React Error Boundary
interface ErrorBoundaryState {
  hasError: boolean;
  error?: Error;
}

class ErrorBoundary extends Component<
  { children: React.ReactNode; fallback?: React.ComponentType<{ error?: Error }> },
  ErrorBoundaryState
> {
  constructor(props: any) {
    super(props);
    this.state = { hasError: false };
  }
  
  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }
  
  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('Error caught by boundary:', error, errorInfo);
    // Send to error reporting service
  }
  
  render() {
    if (this.state.hasError) {
      const FallbackComponent = this.props.fallback || DefaultErrorFallback;
      return <FallbackComponent error={this.state.error} />;
    }
    
    return this.props.children;
  }
}

function DefaultErrorFallback({ error }: { error?: Error }) {
  return (
    <div className="error-fallback">
      <h2>Something went wrong</h2>
      {error && <details>{error.message}</details>}
      <button onClick={() => window.location.reload()}>
        Reload page
      </button>
    </div>
  );
}

// Usage
function App() {
  return (
    <ErrorBoundary>
      <Router>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/dashboard" element={<Dashboard />} />
        </Routes>
      </Router>
    </ErrorBoundary>
  );
}
```

### Async Error Handling

```typescript
// Angular catchError Operator
getData(): Observable<any> {
  return this.http.get('/api/data').pipe(
    catchError(error => {
      console.error('API error:', error);
      return of([]);
    })
  );
}

// React Async Error Handling
type AsyncState<T> = {
  data: T | null;
  loading: boolean;
  error: Error | null;
};

function useAsync<T>(asyncFn: () => Promise<T>): AsyncState<T> {
  const [state, setState] = useState<AsyncState<T>>({
    data: null,
    loading: true,
    error: null
  });
  
  useEffect(() => {
    let cancelled = false;
    
    asyncFn()
      .then(data => {
        if (!cancelled) {
          setState({ data, loading: false, error: null });
        }
      })
      .catch(error => {
        if (!cancelled) {
          setState({ data: null, loading: false, error });
        }
      });
    
    return () => {
      cancelled = true;
    };
  }, [asyncFn]);
  
  return state;
}

// Usage with error boundary
function UserData() {
  const { data: user, loading, error } = useAsync(() => 
    fetch('/api/user/123').then(res => res.json())
  );
  
  if (loading) return <div>Loading...</div>;
  if (error) throw error; // Let error boundary catch it
  
  return <div>Hello, {user?.name}</div>;
}
```

### Debugging Tools

```typescript
// React DevTools Profiler
function App() {
  return (
    <React.Profiler id="App" onRender={onRenderCallback}>
      <UserProfile />
      <UserSettings />
    </React.Profiler>
  );
}

function onRenderCallback(
  id: string, 
  phase: 'mount' | 'update', 
  actualDuration: number, 
  baseDuration: number, 
  startTime: number, 
  commitTime: number
) {
  console.log('Component render:', {
    id,
    phase,
    duration: actualDuration,
    timestamp: new Date().toISOString()
  });
}

// Custom debugging hook
function useDebug(value: any, label?: string) {
  useEffect(() => {
    if (process.env.NODE_ENV === 'development') {
      console.log(`${label || 'Debug'}:`, value);
    }
  }, [value, label]);
  
  return value;
}

// Usage
function Component({ data }: { data: any }) {
  const debugData = useDebug(data, 'Component Data');
  
  return <div>{JSON.stringify(debugData)}</div>;
}

// Performance monitoring
function usePerformanceMonitor(componentName: string) {
  useEffect(() => {
    const startTime = performance.now();
    
    return () => {
      const endTime = performance.now();
      console.log(`${componentName} render time:`, endTime - startTime, 'ms');
    };
  });
}
```

### State Debugging

```typescript
// Redux DevTools-like state logging
function useStateWithLogger<T>(initialState: T, label?: string) {
  const [state, setState] = useState(initialState);
  
  const setStateWithLogger = useCallback((newState: T | ((prev: T) => T)) => {
    setState(prevState => {
      const updatedState = typeof newState === 'function' 
        ? (newState as (prev: T) => T)(prevState)
        : newState;
      
      if (process.env.NODE_ENV === 'development') {
        console.log(`${label || 'State'} update:`, {
          from: prevState,
          to: updatedState,
          timestamp: new Date().toISOString()
        });
      }
      
      return updatedState;
    });
  }, [label]);
  
  return [state, setStateWithLogger] as const;
}

// Usage
function Counter() {
  const [count, setCount] = useStateWithLogger(0, 'Counter');
  
  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={() => setCount(count + 1)}>Increment</button>
    </div>
  );
}
```

## Security Best Practices

### XSS Prevention

**Angular Equivalent**: Built-in DOM sanitization

```typescript
// Angular automatically sanitizes content
// Dangerous content requires bypassing security explicitly
@Component({
  template: `
    <div [innerHTML]="trustedHtml"></div>
    <div>{{ unsafeContent }}</div> <!-- Automatically escaped -->
  `
})
export class SafeComponent {
  trustedHtml = this.sanitizer.bypassSecurityTrustHtml('<script>alert("xss")</script>');
  unsafeContent = '<script>alert("xss")</script>'; <!-- Escaped by Angular -->
}

// React XSS Prevention
function SafeComponent() {
  const dangerousHtml = '<script>alert("xss")</script>';
  
  // Automatically escaped - safe
  const safeDisplay = <div>{dangerousHtml}</div>;
  
  // Dangerous - requires sanitization
  const dangerousDisplay = <div dangerouslySetInnerHTML={{ __html: dangerousHtml }} />;
  
  return (
    <div>
      {safeDisplay}
      {/* Use DOMPurify for sanitization if needed */}
      <div dangerouslySetInnerHTML={{ 
        __html: DOMPurify.sanitize(dangerousHtml) 
      }} />
    </div>
  );
}

// Safe HTML rendering utility
import DOMPurify from 'dompurify';

type SafeHtmlProps = {
  html: string;
  tag?: string;
};

function SafeHtml({ html, tag = 'div' }: SafeHtmlProps) {
  const cleanHtml = useMemo(() => DOMPurify.sanitize(html), [html]);
  const Tag = tag as keyof JSX.IntrinsicElements;
  
  return <Tag dangerouslySetInnerHTML={{ __html: cleanHtml }} />;
}
```

### CSRF Protection

**Angular Equivalent**: Built-in CSRF token handling

```typescript
// Angular HttpClient includes CSRF tokens automatically

// React CSRF Implementation
function useCsrfToken() {
  const [token, setToken] = useState<string>('');
  
  useEffect(() => {
    // Get token from meta tag or cookie
    const metaToken = document.querySelector('meta[name="csrf-token"]')?.getAttribute('content');
    const cookieToken = document.cookie
      .split('; ')
      .find(row => row.startsWith('csrf-token='))
      ?.split('=')[1];
    
    setToken(metaToken || cookieToken || '');
  }, []);
  
  return token;
}

// Secure API client
function useSecureApi() {
  const csrfToken = useCsrfToken();
  
  const secureFetch = useCallback(async (url: string, options: RequestInit = {}) => {
    const secureOptions: RequestInit = {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        'X-CSRF-Token': csrfToken,
        ...options.headers,
      },
      credentials: 'include', // Include cookies
    };
    
    return fetch(url, secureOptions);
  }, [csrfToken]);
  
  return secureFetch;
}

// Usage
function SecureForm() {
  const secureFetch = useSecureApi();
  
  const handleSubmit = async (data: FormData) => {
    try {
      const response = await secureFetch('/api/submit', {
        method: 'POST',
        body: JSON.stringify(data),
      });
      
      if (!response.ok) throw new Error('Request failed');
      
      return await response.json();
    } catch (error) {
      console.error('Secure request failed:', error);
    }
  };
  
  return <form onSubmit={handleSubmit}>{/* form fields */}</form>;
}
```

### Authentication & Authorization

**Angular Equivalent**: Route guards, interceptors

```typescript
// Angular Route Guard
@Injectable()
export class AuthGuard implements CanActivate {
  canActivate(): boolean {
    return this.authService.isLoggedIn();
  }
}

// React Authentication
type AuthContextType = {
  user: User | null;
  login: (credentials: Credentials) => Promise<void>;
  logout: () => void;
  loading: boolean;
};

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    checkAuthStatus();
  }, []);
  
  const checkAuthStatus = async () => {
    try {
      const token = localStorage.getItem('auth_token');
      if (token) {
        const userData = await validateToken(token);
        setUser(userData);
      }
    } catch (error) {
      console.error('Auth check failed:', error);
    } finally {
      setLoading(false);
    }
  };
  
  const login = async (credentials: Credentials) => {
    setLoading(true);
    try {
      const response = await fetch('/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(credentials),
      });
      
      if (!response.ok) throw new Error('Login failed');
      
      const { user, token } = await response.json();
      localStorage.setItem('auth_token', token);
      setUser(user);
    } catch (error) {
      throw error;
    } finally {
      setLoading(false);
    }
  };
  
  const logout = () => {
    localStorage.removeItem('auth_token');
    setUser(null);
  };
  
  return (
    <AuthContext.Provider value={{ user, login, logout, loading }}>
      {children}
    </AuthContext.Provider>
  );
}

// Protected Route Component
type ProtectedRouteProps = {
  children: React.ReactNode;
  requiredRole?: string;
  redirectTo?: string;
};

function ProtectedRoute({ 
  children, 
  requiredRole, 
  redirectTo = '/login' 
}: ProtectedRouteProps) {
  const { user, loading } = useAuth();
  const location = useLocation();
  
  if (loading) return <div>Loading...</div>;
  
  if (!user) {
    return <Navigate to={redirectTo} state={{ from: location }} replace />;
  }
  
  if (requiredRole && !user.roles.includes(requiredRole)) {
    return <Navigate to="/unauthorized" replace />;
  }
  
  return <>{children}</>;
}

// Usage
function App() {
  return (
    <AuthProvider>
      <Router>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route 
            path="/dashboard" 
            element={
              <ProtectedRoute>
                <Dashboard />
              </ProtectedRoute>
            } 
          />
          <Route 
            path="/admin" 
            element={
              <ProtectedRoute requiredRole="admin">
                <AdminPanel />
              </ProtectedRoute>
            } 
          />
        </Routes>
      </Router>
    </AuthProvider>
  );
}
```

### Secure Data Handling

```typescript
// Angular HttpClient with interceptors for security
// React secure data handling utilities

// Environment-specific configuration
const config = {
  apiUrl: process.env.REACT_APP_API_URL,
  isDevelopment: process.env.NODE_ENV === 'development',
};

// Secure localStorage wrapper
const secureStorage = {
  setItem: (key: string, value: string) => {
    if (typeof window !== 'undefined') {
      localStorage.setItem(key, value);
    }
  },
  
  getItem: (key: string): string | null => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem(key);
    }
    return null;
  },
  
  removeItem: (key: string) => {
    if (typeof window !== 'undefined') {
      localStorage.removeItem(key);
    }
  },
};

// Sensitive data validation
type SensitiveData = {
  email: string;
  password: string;
  ssn?: string;
};

function validateSensitiveData(data: Partial<SensitiveData>): boolean {
  if (data.email && !isValidEmail(data.email)) {
    return false;
  }
  
  if (data.password && data.password.length < 8) {
    return false;
  }
  
  if (data.ssn && !isValidSSN(data.ssn)) {
    return false;
  }
  
  return true;
}

// Input sanitization
function sanitizeInput(input: string): string {
  return input
    .replace(/[<>]/g, '') // Remove potential HTML
    .trim()
    .substring(0, 1000); // Limit length
}

function SecureForm() {
  const [formData, setFormData] = useState({
    email: '',
    message: '',
  });
  
  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: sanitizeInput(value),
    }));
  };
  
  return (
    <form>
      <input
        type="email"
        name="email"
        value={formData.email}
        onChange={handleChange}
        required
      />
      <textarea
        name="message"
        value={formData.message}
        onChange={handleChange}
        maxLength={1000}
      />
    </form>
  );
}
```

## Migration Guide

### Step-by-Step Migration Process

#### Phase 1: Setup and Foundation

1. **Create React Project Structure**
```bash
# Create new React project with Vite (recommended)
npm create vite@latest my-react-app -- --template react-ts

# Install essential dependencies
npm install react-router-dom axios
npm install -D @testing-library/react @testing-library/jest-dom

# Alternative: Next.js for full-stack
npm create next-app@latest my-app --typescript --app
```

2. **Set up TypeScript Configuration**
```json
{
  "compilerOptions": {
    "target": "es5",
    "lib": ["dom", "dom.iterable", "es6"],
    "allowJs": true,
    "skipLibCheck": true,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "forceConsistentCasingInFileNames": true,
    "moduleResolution": "node",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx"
  },
  "include": ["src"]
}
```

#### Phase 2: Convert Components

**Angular Service to React Hook**
```typescript
// Angular Service (Before)
@Injectable()
export class UserService {
  private users$ = new BehaviorSubject<User[]>([]);
  
  constructor(private http: HttpClient) {
    this.loadUsers();
  }
  
  get users(): Observable<User[]> {
    return this.users$.asObservable();
  }
  
  addUser(user: User) {
    const current = this.users$.value;
    this.users$.next([...current, user]);
  }
}

// React Hook (After)
type UseUsersReturn = {
  users: User[];
  loading: boolean;
  error: Error | null;
  addUser: (user: Omit<User, 'id'>) => void;
};

function useUsers(): UseUsersReturn {
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [isPending, startTransition] = useTransition();
  
  useEffect(() => {
    loadUsers();
  }, []);
  
  const loadUsers = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/users');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data: User[] = await response.json();
      
      startTransition(() => {
        setUsers(data);
      });
    } catch (err) {
      setError(err as Error);
    } finally {
      setLoading(false);
    }
  }, []);
  
  const addUser = useCallback(async (userData: Omit<User, 'id'>) => {
    try {
      const response = await fetch('/api/users', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(userData),
      });
      
      if (!response.ok) {
        throw new Error(`Failed to create user: ${response.status}`);
      }
      
      const newUser: User = await response.json();
      
      startTransition(() => {
        setUsers(prev => [...prev, newUser]);
      });
      
      return newUser;
    } catch (err) {
      setError(err as Error);
      throw err;
    }
  }, []);
  
  return { users, loading, error, addUser, isAdding: isPending };
}
```

#### Phase 3: Routing Migration

```typescript
// Angular Routing (Before)
const routes: Routes = [
  { path: '', component: HomeComponent },
  { path: 'users', component: UserListComponent },
  { path: 'users/:id', component: UserDetailComponent },
  { path: 'admin', component: AdminComponent, canActivate: [AuthGuard] }
];

// React Routing (After)
function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/users" element={<UserList />} />
        <Route path="/users/:id" element={<UserDetail />} />
        <Route 
          path="/admin" 
          element={
            <ProtectedRoute>
              <Admin />
            </ProtectedRoute>
          } 
        />
      </Routes>
    </BrowserRouter>
  );
}
```

### Migration Checklist

#### Pre-Migration
- [ ] Inventory all Angular components, services, and modules
- [ ] Identify external dependencies and find React equivalents
- [ ] Set up version control branch for migration
- [ ] Create a migration plan with priorities and timelines

#### Migration Steps
- [ ] Set up React project structure
- [ ] Convert TypeScript interfaces and types
- [ ] Migrate services to custom hooks
- [ ] Convert components one by one
- [ ] Set up routing
- [ ] Migrate forms
- [ ] Add error boundaries
- [ ] Implement testing
- [ ] Add performance optimizations

#### Post-Migration
- [ ] Comprehensive testing
- [ ] Performance benchmarking
- [ ] Accessibility audit
- [ ] Security review
- [ ] Documentation updates
- [ ] Team training
- [ ] Production deployment strategy

```

## Modern React Ecosystem

### Next.js (React Framework)

**Angular Equivalent**: Angular Universal + Angular CLI

```typescript
// Next.js App Router (Modern)
// app/dashboard/page.tsx
import { getUser } from '@/lib/auth';
import { redirect } from 'next/navigation';

async function DashboardPage() {
  const user = await getUser();
  
  if (!user) {
    redirect('/login');
  }
  
  return (
    <div>
      <h1>Welcome, {user.name}</h1>
      <DashboardContent userId={user.id} />
    </div>
  );
}

// Server Components by default
async function DashboardContent({ userId }: { userId: string }) {
  const data = await fetchData(userId);
  
  return (
    <div>
      <h2>Your Data</h2>
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </div>
  );
}

// Client Components for interactivity
'use client';

import { useState } from 'react';

function InteractiveComponent() {
  const [count, setCount] = useState(0);
  
  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={() => setCount(count + 1)}>Increment</button>
    </div>
  );
}
```

### Remix (Full-stack React Framework)

**Angular Equivalent**: Angular with integrated backend

```typescript
// Remix Route
// app/routes/users.tsx
import { json } from '@remix-run/node';
import { useLoaderData } from '@remix-run/react';

// Server-side data loading
export async function loader() {
  const users = await fetchUsers();
  return json({ users });
}

// Server-side action handling
export async function action({ request }: ActionFunctionArgs) {
  const formData = await request.formData();
  const name = formData.get('name');
  
  const newUser = await createUser({ name });
  return json(newUser);
}

export default function UsersRoute() {
  const { users } = useLoaderData<typeof loader>();
  
  return (
    <div>
      <h1>Users</h1>
      <UserList users={users} />
      <AddUserForm />
    </div>
  );
}

// Form with Remix Form component
import { Form } from '@remix-run/react';

function AddUserForm() {
  return (
    <Form method="post">
      <input name="name" placeholder="User name" required />
      <button type="submit">Add User</button>
    </Form>
  );
}
```

### State Management Libraries

#### Redux Toolkit (Modern Redux)

**Angular Equivalent**: NgRx Store

```typescript
// Redux Toolkit Setup
import { configureStore, createSlice, createAsyncThunk } from '@reduxjs/toolkit';

// Async Thunk
export const fetchUsers = createAsyncThunk(
  'users/fetchUsers',
  async () => {
    const response = await fetch('/api/users');
    return response.json();
  }
);

// Slice
const usersSlice = createSlice({
  name: 'users',
  initialState: {
    users: [],
    status: 'idle',
    error: null,
  },
  reducers: {},
  extraReducers: (builder) => {
    builder
      .addCase(fetchUsers.pending, (state) => {
        state.status = 'loading';
      })
      .addCase(fetchUsers.fulfilled, (state, action) => {
        state.status = 'succeeded';
        state.users = action.payload;
      })
      .addCase(fetchUsers.rejected, (state, action) => {
        state.status = 'failed';
        state.error = action.error.message;
      });
  },
});

// Store Configuration
const store = configureStore({
  reducer: {
    users: usersSlice.reducer,
  },
});

// Component Usage
import { useSelector, useDispatch } from 'react-redux';

function UserList() {
  const { users, status, error } = useSelector((state) => state.users);
  const dispatch = useDispatch();
  
  useEffect(() => {
    if (status === 'idle') {
      dispatch(fetchUsers());
    }
  }, [status, dispatch]);
  
  if (status === 'loading') return <div>Loading...</div>;
  if (status === 'failed') return <div>Error: {error}</div>;
  
  return (
    <ul>
      {users.map(user => (
        <li key={user.id}>{user.name}</li>
      ))}
    </ul>
  );
}
```

#### Zustand (Simpler State Management)

**Angular Equivalent**: Simple services with BehaviorSubject

```typescript
// Zustand Store
import { create } from 'zustand';

type UserStore = {
  users: User[];
  loading: boolean;
  fetchUsers: () => Promise<void>;
  addUser: (user: Omit<User, 'id'>) => void;
};

const useUserStore = create<UserStore>((set, get) => ({
  users: [],
  loading: false,
  
  fetchUsers: async () => {
    set({ loading: true });
    try {
      const response = await fetch('/api/users');
      const users = await response.json();
      set({ users, loading: false });
    } catch (error) {
      console.error('Failed to fetch users:', error);
      set({ loading: false });
    }
  },
  
  addUser: (userData) => {
    const newUser = { ...userData, id: Date.now().toString() };
    set(state => ({ users: [...state.users, newUser] }));
  },
}));

// Component Usage
function UserList() {
  const { users, loading, fetchUsers } = useUserStore();
  
  useEffect(() => {
    fetchUsers();
  }, [fetchUsers]);
  
  if (loading) return <div>Loading...</div>;
  
  return (
    <ul>
      {users.map(user => (
        <li key={user.id}>{user.name}</li>
      ))}
    </ul>
  );
}
```

### UI Component Libraries

#### Material-UI (MUI)

**Angular Equivalent**: Angular Material

```typescript
// MUI Components
import {
  Box,
  Card,
  CardContent,
  CardActions,
  Button,
  TextField,
  Typography,
  Container,
} from '@mui/material';

function UserCard({ user, onEdit, onDelete }: {
  user: User;
  onEdit: (user: User) => void;
  onDelete: (id: string) => void;
}) {
  return (
    <Card sx={{ m: 2, maxWidth: 300 }}>
      <CardContent>
        <Typography variant="h6">{user.name}</Typography>
        <Typography color="text.secondary">{user.email}</Typography>
      </CardContent>
      <CardActions>
        <Button size="small" onClick={() => onEdit(user)}>
          Edit
        </Button>
        <Button size="small" color="error" onClick={() => onDelete(user.id)}>
          Delete
        </Button>
      </CardActions>
    </Card>
  );
}

function UserManagement() {
  const [users, setUsers] = useState<User[]>([]);
  const [formData, setFormData] = useState({ name: '', email: '' });
  
  return (
    <Container maxWidth="md">
      <Typography variant="h4" gutterBottom>
        User Management
      </Typography>
      
      <Box component="form" sx={{ mb: 3 }}>
        <TextField
          fullWidth
          label="Name"
          value={formData.name}
          onChange={(e) => setFormData({ ...formData, name: e.target.value })}
          margin="normal"
        />
        <TextField
          fullWidth
          label="Email"
          type="email"
          value={formData.email}
          onChange={(e) => setFormData({ ...formData, email: e.target.value })}
          margin="normal"
        />
        <Button variant="contained" sx={{ mt: 2 }}>
          Add User
        </Button>
      </Box>
      
      <Box display="flex" flexWrap="wrap" gap={2}>
        {users.map(user => (
          <UserCard
            key={user.id}
            user={user}
            onEdit={(user) => console.log('Edit:', user)}
            onDelete={(id) => setUsers(users.filter(u => u.id !== id))}
          />
        ))}
      </Box>
    </Container>
  );
}
```

#### Chakra UI

```typescript
// Chakra UI Components
import {
  Box,
  Card,
  CardHeader,
  CardBody,
  Button,
  Input,
  Stack,
  Heading,
  Text,
  Container,
  useToast,
} from '@chakra-ui/react';

function UserForm({ onUserAdded }: { onUserAdded: (user: User) => void }) {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const toast = useToast();
  
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!name || !email) {
      toast({
        title: 'Error',
        description: 'Please fill all fields',
        status: 'error',
        duration: 3000,
      });
      return;
    }
    
    const newUser = { id: Date.now().toString(), name, email };
    onUserAdded(newUser);
    
    toast({
      title: 'Success',
      description: 'User added successfully',
      status: 'success',
      duration: 3000,
    });
    
    setName('');
    setEmail('');
  };
  
  return (
    <Card>
      <CardHeader>
        <Heading size="md">Add New User</Heading>
      </CardHeader>
      <CardBody>
        <form onSubmit={handleSubmit}>
          <Stack spacing={3}>
            <Input
              placeholder="User name"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
            <Input
              type="email"
              placeholder="Email address"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
            <Button type="submit" colorScheme="blue">
              Add User
            </Button>
          </Stack>
        </form>
      </CardBody>
    </Card>
  );
}
```

### Data Fetching Libraries

#### TanStack Query (React Query)

**Angular Equivalent**: HTTP Interceptors with caching

```typescript
// TanStack Query Setup
import { QueryClient, QueryClientProvider, useQuery, useMutation } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <UserManagement />
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  );
}

// Custom Hook for data fetching
function useUsers() {
  return useQuery({
    queryKey: ['users'],
    queryFn: async () => {
      const response = await fetch('/api/users');
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      return response.json();
    },
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

function useCreateUser() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (userData: Omit<User, 'id'>) => {
      const response = await fetch('/api/users', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(userData),
      });
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['users'] });
    },
  });
}

// Component with TanStack Query
function UserList() {
  const { data: users, error, isLoading } = useUsers();
  const createUser = useCreateUser();
  
  if (isLoading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;
  
  return (
    <div>
      <h2>Users</h2>
      <ul>
        {users?.map(user => (
          <li key={user.id}>{user.name}</li>
        ))}
      </ul>
      <button onClick={() => createUser.mutate({ name: 'New User', email: 'new@example.com' })}>
        Add User
      </button>
    </div>
  );
}
```

## Common Pitfalls & Best Practices

### Common Mistakes for Angular Developers

#### 1. Treating Components Like Angular Controllers

❌ **Wrong Approach**:
```typescript
// Trying to use component class methods like Angular
class UserComponent {
  users: User[] = [];
  
  constructor() {
    this.loadUsers(); // Wrong - no lifecycle hooks in function components
  }
  
  loadUsers() {
    // This won't work as expected
  }
}
```

✅ **Correct Approach**:
```typescript
function UserComponent() {
  const [users, setUsers] = useState<User[]>([]);
  
  useEffect(() => {
    loadUsers();
  }, []); // Correct dependency array
  
  const loadUsers = async () => {
    const data = await fetchUsers();
    setUsers(data);
  };
  
  return <div>{/* JSX */}</div>;
}
```

#### 2. Missing Keys in Lists

❌ **Wrong**:
```typescript
{users.map(user => (
  <div>{user.name}</div> // Missing key prop
))}
```

✅ **Correct**:
```typescript
{users.map(user => (
  <div key={user.id}>{user.name}</div> // Always include stable key
))}
```

#### 3. Direct State Mutation

❌ **Wrong**:
```typescript
const [users, setUsers] = useState<User[]>([]);

const addUser = (newUser: User) => {
  users.push(newUser); // Direct mutation
  setUsers(users); // Won't trigger re-render
};
```

✅ **Correct**:
```typescript
const addUser = (newUser: User) => {
  setUsers(prev => [...prev, newUser]); // Immutable update
};
```

#### 4. Incorrect useEffect Dependencies

❌ **Wrong**:
```typescript
useEffect(() => {
  fetchData(userId); // Using userId but not in dependencies
}, []); // Missing userId dependency
```

✅ **Correct**:
```typescript
useEffect(() => {
  fetchData(userId);
}, [userId]); // Include all dependencies
```

#### 5. Creating Functions in Render

❌ **Wrong**:
```typescript
function UserList({ users }: { users: User[] }) {
  return (
    <div>
      {users.map(user => (
        <UserCard 
          key={user.id} 
          user={user}
          onClick={() => setSelectedUser(user)} // New function on every render
        />
      ))}
    </div>
  );
}
```

✅ **Correct**:
```typescript
function UserList({ users }: { users: User[] }) {
  const handleUserClick = useCallback((user: User) => {
    setSelectedUser(user);
  }, []);
  
  return (
    <div>
      {users.map(user => (
        <UserCard 
          key={user.id} 
          user={user}
          onClick={() => handleUserClick(user)}
        />
      ))}
    </div>
  );
}
```

### Best Practices

#### 1. Component Design Patterns

```typescript
// Prefer composition over inheritance
interface ButtonProps {
  variant: 'primary' | 'secondary';
  size: 'small' | 'medium' | 'large';
  children: React.ReactNode;
  onClick?: () => void;
}

function Button({ variant = 'primary', size = 'medium', children, onClick }: ButtonProps) {
  const baseClasses = 'btn';
  const variantClasses = `btn-${variant}`;
  const sizeClasses = `btn-${size}`;
  
  const className = `${baseClasses} ${variantClasses} ${sizeClasses}`;
  
  return (
    <button className={className} onClick={onClick}>
      {children}
    </button>
  );
}

// Usage with composition
function ActionButton({ children, ...props }: ButtonProps) {
  return (
    <div className="action-button-wrapper">
      <Button {...props}>{children}</Button>
    </div>
  );
}
```

#### 2. Custom Hooks for Logic Reuse

```typescript
// Good: Extract business logic into custom hooks
function useLocalStorage<T>(key: string, initialValue: T) {
  const [storedValue, setStoredValue] = useState<T>(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      console.error(`Error reading localStorage key "${key}":`, error);
      return initialValue;
    }
  });
  
  const setValue = useCallback((value: T | ((val: T) => T)) => {
    try {
      const valueToStore = value instanceof Function ? value(storedValue) : value;
      setStoredValue(valueToStore);
      window.localStorage.setItem(key, JSON.stringify(valueToStore));
    } catch (error) {
      console.error(`Error setting localStorage key "${key}":`, error);
    }
  }, [key, storedValue]);
  
  return [storedValue, setValue] as const;
}

// Usage
function SettingsComponent() {
  const [theme, setTheme] = useLocalStorage<'light' | 'dark'>('theme', 'light');
  const [language, setLanguage] = useLocalStorage<string>('language', 'en');
  
  return (
    <div>
      <select value={theme} onChange={(e) => setTheme(e.target.value as 'light' | 'dark')}>
        <option value="light">Light</option>
        <option value="dark">Dark</option>
      </select>
    </div>
  );
}
```

#### 3. Error Boundaries for Production

```typescript
// Wrap your app with error boundaries
function App() {
  return (
    <ErrorBoundary fallback={<ErrorFallback />}>
      <Router>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/users" element={
            <ErrorBoundary fallback={<UsersErrorFallback />}>
              <Users />
            </ErrorBoundary>
          } />
        </Routes>
      </Router>
    </ErrorBoundary>
  );
}

function UsersErrorFallback() {
  return (
    <div>
      <h2>Unable to load users</h2>
      <p>There was a problem loading the users list.</p>
      <button onClick={() => window.location.reload()}>
        Try again
      </button>
    </div>
  );
}
```

#### 4. Performance Optimization Patterns

```typescript
// Use React.memo for expensive components
const ExpensiveComponent = React.memo(({ data, onItemClick }: {
  data: Item[];
  onItemClick: (item: Item) => void;
}) => {
  return (
    <div>
      {data.map(item => (
        <div key={item.id} onClick={() => onItemClick(item)}>
          {item.name}
        </div>
      ))}
    </div>
  );
});

// Use useMemo for expensive calculations
function Chart({ data }: { data: DataPoint[] }) {
  const processedData = useMemo(() => {
    return data.map(point => ({
      ...point,
      transformedValue: complexCalculation(point.value)
    }));
  }, [data]);
  
  return <ChartComponent data={processedData} />;
}

// Use useCallback for event handlers
function ParentComponent({ items }: { items: Item[] }) {
  const [selectedItem, setSelectedItem] = useState<Item | null>(null);
  
  const handleItemClick = useCallback((item: Item) => {
    setSelectedItem(item);
  }, []);
  
  return (
    <div>
      {items.map(item => (
        <ItemComponent 
          key={item.id}
          item={item}
          onClick={handleItemClick}
        />
      ))}
    </div>
  );
}
```

### Testing Best Practices

```typescript
// Test component behavior, not implementation
describe('UserCard', () => {
  it('calls onEdit when edit button is clicked', async () => {
    const user = { id: '1', name: 'John', email: 'john@example.com' };
    const onEdit = vi.fn();
    
    render(<UserCard user={user} onEdit={onEdit} />);
    
    const editButton = screen.getByRole('button', { name: /edit/i });
    await userEvent.click(editButton);
    
    expect(onEdit).toHaveBeenCalledWith(user);
  });
  
  it('displays user information correctly', () => {
    const user = { id: '1', name: 'John', email: 'john@example.com' };
    
    render(<UserCard user={user} onEdit={() => {}} />);
    
    expect(screen.getByText('John')).toBeInTheDocument();
    expect(screen.getByText('john@example.com')).toBeInTheDocument();
  });
});
```

## Quick Reference

| Angular Concept | React Equivalent |
|----------------|------------------|
| Standalone Component | Function Component |
| input() signal | Function parameters |
| output() function | Callback props |
| @if/@else block | {condition ? <A /> : <B />} |
| @for block | {array.map(item => <Component key={item.id} />)} |
| @empty block | {array.length === 0 ? <Empty /> : null} |
| signal() state | useState() |
| [(ngModel)] | useState + onChange |
| @Injectable() + inject() | Custom hooks |
| Router (standalone) | React Router v6.4+ |
| HttpClient + toSignal() | fetch/axios + React Query |
| Pipes | JavaScript functions |
| Interceptors | axios interceptors/React Query |
| Guards | Route loaders/actions |
| Modules (deprecated) | No direct equivalent |
| @Input/@Output | Function parameters + callbacks |
| @if/@else block | {condition ? <A /> : <B />} |
| @for block | {array.map(item => <Component key={item.id} />)} |
| @empty block | {array.length === 0 ? <Empty /> : null} |
| signal() + effect() | useState() + useEffect() |
| computed() | useMemo() |
| Change Detection | Re-renders on state change |
| DestroyRef | useEffect cleanup |

## Learning Path (Updated for 2024)

1. **Modern React Setup** - Learn Vite, TypeScript, and modern tooling
2. **Master Hooks** - useState, useEffect, useTransition, useDeferredValue
3. **Understand React 18 Concurrent Features** - startTransition, automatic batching
4. **Learn Modern State Management** - Zustand, RTK Query, or React Query
5. **Master React Router v6.4+** - Data routers, loaders, and actions
6. **Practice Modern Forms** - React Hook Form + Zod validation
7. **Explore Testing** - React Testing Library v5+ with async patterns
8. **Build Production Apps** - Performance optimization, error boundaries, SSR/SSG
9. **Study Modern Ecosystem** - Next.js 14+, Remix, or modern frameworks
10. **Learn Performance** - Memoization, code splitting, concurrent rendering

## Resources (Updated for 2024)

### Official Documentation
- [Official React Documentation](https://react.dev)
- [Angular Documentation](https://angular.dev)
- [React Tutorial for Angular Developers](https://react.dev/learn)

### Modern React Ecosystem
- [Vite Documentation](https://vitejs.dev)
- [React Router v6.4+](https://reactrouter.com)
- [React Hook Form](https://react-hook-form.com)
- [Zod Validation](https://zod.dev)
- [Zustand State Management](https://zustand.docs.pmnd.rs)
- [TanStack Query (React Query)](https://tanstack.com/query)
- [Next.js 14+](https://nextjs.org/docs)
- [Remix](https://remix.run/docs)

### Modern Angular (for comparison)
- [Angular Standalone Components](https://angular.dev/guide/standalone-components)
- [Angular Signals](https://angular.dev/guide/signals)
- [Angular 17+ New Features](https://angular.dev/update-guide)

### Testing
- [React Testing Library](https://testing-library.com/docs/react-testing-library/intro)
- [Testing Library Angular Guide](https://testing-library.com/docs/angular-testing-library/intro)
- [Vitest](https://vitest.dev)

### Comparisons & Best Practices
- [React vs Angular 2024](https://www.robinwieruch.de/react-vs-angular)
- [Modern React Patterns](https://reactpatterns.com)
- [Angular to React Migration Guide](https://www.angularmigrations.com)
- [React Performance Guide](https://kentcdodds.com/blog/profile-a-react-app-for-performance)
- [Next.js Documentation](https://nextjs.org/docs)
- [React Testing Library](https://testing-library.com/docs/react-testing-library/intro/)
- [TypeScript React Cheatsheet](https://react-typescript-cheatsheet.netlify.app/)
- [React Performance Guide](https://kentcdodds.com/blog/profile-a-react-app-for-performance)

## Real-World Examples

### Complete Application Structure

```typescript
// app/types/index.ts
export interface User {
  id: string;
  name: string;
  email: string;
  avatar?: string;
  createdAt: string;
  updatedAt: string;
}

export interface ApiResponse<T> {
  data: T;
  message: string;
  success: boolean;
}

// app/hooks/useApi.ts
export function useApi() {
  const baseUrl = process.env.REACT_APP_API_URL || 'http://localhost:3001/api';
  
  const request = useCallback(async <T>(endpoint: string, options: RequestInit = {}): Promise<T> => {
    const url = `${baseUrl}${endpoint}`;
    const config: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };
    
    const response = await fetch(url, config);
    
    if (!response.ok) {
      throw new Error(`API Error: ${response.statusText}`);
    }
    
    return response.json();
  }, [baseUrl]);
  
  return { request };
}

// app/hooks/useUsers.ts
export function useUsers() {
  const { request } = useApi();
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  
  const fetchUsers = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await request<ApiResponse<User[]>>('/users');
      setUsers(response.data);
    } catch (err) {
      setError(err as Error);
    } finally {
      setLoading(false);
    }
  }, [request]);
  
  const createUser = useCallback(async (userData: Omit<User, 'id' | 'createdAt' | 'updatedAt'>) => {
    try {
      const response = await request<ApiResponse<User>>('/users', {
        method: 'POST',
        body: JSON.stringify(userData),
      });
      
      setUsers(prev => [...prev, response.data]);
      return response.data;
    } catch (err) {
      throw err;
    }
  }, [request]);
  
  const updateUser = useCallback(async (id: string, userData: Partial<User>) => {
    try {
      const response = await request<ApiResponse<User>>(`/users/${id}`, {
        method: 'PUT',
        body: JSON.stringify(userData),
      });
      
      setUsers(prev => prev.map(user => 
        user.id === id ? response.data : user
      ));
      
      return response.data;
    } catch (err) {
      throw err;
    }
  }, [request]);
  
  const deleteUser = useCallback(async (id: string) => {
    try {
      await request(`/users/${id}`, { method: 'DELETE' });
      setUsers(prev => prev.filter(user => user.id !== id));
    } catch (err) {
      throw err;
    }
  }, [request]);
  
  useEffect(() => {
    fetchUsers();
  }, [fetchUsers]);
  
  return {
    users,
    loading,
    error,
    refetch: fetchUsers,
    createUser,
    updateUser,
    deleteUser,
  };
}

// app/components/UserForm.tsx
interface UserFormProps {
  user?: User;
  onSubmit: (userData: Partial<User>) => Promise<void>;
  onCancel: () => void;
  loading?: boolean;
}

export function UserForm({ user, onSubmit, onCancel, loading }: UserFormProps) {
  const [formData, setFormData] = useState({
    name: user?.name || '',
    email: user?.email || '',
  });
  const [errors, setErrors] = useState<Record<string, string>>({});
  
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    
    // Clear error when user starts typing
    if (errors[name]) {
      setErrors(prev => ({ ...prev, [name]: '' }));
    }
  };
  
  const validateForm = () => {
    const newErrors: Record<string, string> = {};
    
    if (!formData.name.trim()) {
      newErrors.name = 'Name is required';
    }
    
    if (!formData.email.trim()) {
      newErrors.email = 'Email is required';
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) {
      newErrors.email = 'Invalid email format';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }
    
    try {
      await onSubmit(formData);
    } catch (error) {
      console.error('Form submission failed:', error);
    }
  };
  
  return (
    <form onSubmit={handleSubmit} className="user-form">
      <div className="form-group">
        <label htmlFor="name">Name</label>
        <input
          id="name"
          name="name"
          type="text"
          value={formData.name}
          onChange={handleChange}
          className={errors.name ? 'error' : ''}
          disabled={loading}
        />
        {errors.name && <span className="error-message">{errors.name}</span>}
      </div>
      
      <div className="form-group">
        <label htmlFor="email">Email</label>
        <input
          id="email"
          name="email"
          type="email"
          value={formData.email}
          onChange={handleChange}
          className={errors.email ? 'error' : ''}
          disabled={loading}
        />
        {errors.email && <span className="error-message">{errors.email}</span>}
      </div>
      
      <div className="form-actions">
        <button type="button" onClick={onCancel} disabled={loading}>
          Cancel
        </button>
        <button type="submit" disabled={loading}>
          {loading ? 'Saving...' : user ? 'Update' : 'Create'}
        </button>
      </div>
    </form>
  );
}

// app/components/UserList.tsx
export function UserList() {
  const { users, loading, error, createUser, updateUser, deleteUser } = useUsers();
  const [editingUser, setEditingUser] = useState<User | null>(null);
  const [isCreating, setIsCreating] = useState(false);
  const [submitLoading, setSubmitLoading] = useState(false);
  
  const handleCreateUser = async (userData: Partial<User>) => {
    setSubmitLoading(true);
    try {
      await createUser(userData as Omit<User, 'id' | 'createdAt' | 'updatedAt'>);
      setIsCreating(false);
    } catch (error) {
      console.error('Failed to create user:', error);
    } finally {
      setSubmitLoading(false);
    }
  };
  
  const handleUpdateUser = async (userData: Partial<User>) => {
    if (!editingUser) return;
    
    setSubmitLoading(true);
    try {
      await updateUser(editingUser.id, userData);
      setEditingUser(null);
    } catch (error) {
      console.error('Failed to update user:', error);
    } finally {
      setSubmitLoading(false);
    }
  };
  
  const handleDeleteUser = async (id: string) => {
    if (!confirm('Are you sure you want to delete this user?')) {
      return;
    }
    
    try {
      await deleteUser(id);
    } catch (error) {
      console.error('Failed to delete user:', error);
    }
  };
  
  if (loading && users.length === 0) {
    return <div className="loading">Loading users...</div>;
  }
  
  if (error) {
    return (
      <div className="error">
        <p>Failed to load users: {error.message}</p>
        <button onClick={() => window.location.reload()}>Retry</button>
      </div>
    );
  }
  
  return (
    <div className="user-list">
      <div className="user-list-header">
        <h1>User Management</h1>
        <button 
          onClick={() => setIsCreating(true)}
          className="btn btn-primary"
        >
          Add New User
        </button>
      </div>
      
      {(isCreating || editingUser) && (
        <div className="modal">
          <div className="modal-content">
            <h2>{editingUser ? 'Edit User' : 'Create User'}</h2>
            <UserForm
              user={editingUser || undefined}
              onSubmit={editingUser ? handleUpdateUser : handleCreateUser}
              onCancel={() => {
                setIsCreating(false);
                setEditingUser(null);
              }}
              loading={submitLoading}
            />
          </div>
        </div>
      )}
      
      {users.length === 0 ? (
        <div className="empty-state">
          <p>No users found. Create your first user!</p>
        </div>
      ) : (
        <div className="user-grid">
          {users.map(user => (
            <div key={user.id} className="user-card">
              <div className="user-avatar">
                {user.avatar ? (
                  <img src={user.avatar} alt={user.name} />
                ) : (
                  <div className="avatar-placeholder">
                    {user.name.charAt(0).toUpperCase()}
                  </div>
                )}
              </div>
              
              <div className="user-info">
                <h3>{user.name}</h3>
                <p>{user.email}</p>
                <small>Created: {new Date(user.createdAt).toLocaleDateString()}</small>
              </div>
              
              <div className="user-actions">
                <button 
                  onClick={() => setEditingUser(user)}
                  className="btn btn-secondary"
                >
                  Edit
                </button>
                <button 
                  onClick={() => handleDeleteUser(user.id)}
                  className="btn btn-danger"
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// App.tsx
function App() {
  return (
    <ErrorBoundary>
      <Router>
        <div className="app">
          <header className="app-header">
            <nav>
              <Link to="/">Home</Link>
              <Link to="/users">Users</Link>
            </nav>
          </header>
          
          <main className="app-main">
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/users" element={<UserList />} />
            </Routes>
          </main>
        </div>
      </Router>
    </ErrorBoundary>
  );
}

export default App;
```

## ✅ **Updated to Latest Standards (2024)**

### **Angular 17+ Features Added:**
- ✅ **Standalone Components** - All examples use standalone approach
- ✅ **Angular Signals** - Modern signal-based state management
- ✅ **Modern Input/Output** - `input()` and `output()` functions alongside decorators
- ✅ **Modern RxJS Interop** - `toSignal()` and `toObservable()` for seamless integration
- ✅ **Enhanced Change Detection** - Signal-based reactivity with computed values
- ✅ **Modern Dependency Injection** - `inject()` function patterns
- ✅ **@if/@for Blocks** - New control flow syntax
- ✅ **Modern Testing** - Standalone component testing patterns

### **React 18+ Features Added:**
- ✅ **Concurrent Features** - `useTransition`, `useDeferredValue`, automatic batching
- ✅ **Modern Build Tools** - Vite instead of Create React App
- ✅ **React Router v6.4+** - Data routers, loaders, and actions
- ✅ **Modern State Management** - RTK Query, Zustand v4+, React Query v5
- ✅ **TypeScript Best Practices** - Strict typing, utility types, discriminated unions
- ✅ **Modern Forms** - React Hook Form + Zod validation
- ✅ **Testing Library v5+** - Proper async patterns, userEvent v14+
- ✅ **Performance Optimization** - Memoization, code splitting, concurrent rendering

### **Key Modernization Highlights:**
- 🔄 **Angular**: NgModules → Standalone components
- 🔄 **React**: Class components → Function components with hooks
- 🔄 **State**: Subjects/RxJS → Signals/useState
- 🔄 **Build**: Angular CLI/Webpack → Vite/modern bundling
- 🔄 **Testing**: Karma/Jasmine → Modern testing approaches
- 🔄 **Forms**: Legacy patterns → Modern reactive patterns

This ultimate Angular to React comparison document now covers:

✅ **Complete Concept Mapping** - Every Angular concept has its React equivalent
✅ **Advanced Patterns** - HOCs, render props, compound components, custom hooks  
✅ **Performance Optimization** - Memoization, lazy loading, code splitting
✅ **TypeScript Mastery** - Advanced types, generics, discriminated unions
✅ **Error Handling** - Error boundaries, async error handling, debugging
✅ **Security** - XSS prevention, CSRF protection, authentication
✅ **Migration Guide** - Step-by-step process with real examples
✅ **Modern Ecosystem** - Next.js, Remix, state management, UI libraries
✅ **Best Practices** - Common pitfalls and proven patterns
✅ **Real-World Example** - Complete application structure

This is the most comprehensive Angular to React learning resource available, covering everything from basic concepts to production-ready patterns.