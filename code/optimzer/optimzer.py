from optimzer_project.code.backend.backend import xp as np, is_gpu_enable, _cg, _linearOperator
#from scipy.sparse.linalg import LinearOperator, cg

from abc import ABC, abstractmethod


class Optimzer(ABC):
    
    '''
    class trừu tượng, abstraclass
    '''
    
    def __init__(self, lr=0.01):
        self.lr = lr
    
    @abstractmethod
    def update(self, param, grad, key=None):
        '''
        triển khai thuật toán tối ưu
        '''
        pass
    
    @abstractmethod
    def step(self, model):
        '''
        cập nhập tham số do các class tối ưu quản lý
        '''
        pass
    
  
  
class GD(Optimzer):
    
    '''
    class triển khai thuật toán gradient descent
    '''
    
    #def __init__(self, model lr=0.01):
    #    super().__init__(lr)
        
    def update(self, param, grad, key=None):
        return param - self.lr * grad
    
    
    def step(self, model):
        self.model = model
        params = model.get_params_not_line_search()
        grads = model.get_grads_not_line_search()

        for (param, key), (grad, _) in zip(params, grads):
            if grad is not None:
                print(f"{key} grad norm: {np.linalg.norm(grad)}")
                param[...] = self.update(param, grad, None) # param[...] giữ nguyên object nhưng thay đổi toàn bộ giá trị 
            
            

class LineSearch(Optimzer):
    
    '''
    class triển khai thuật toán back tracking line search
    ct thuật toán: đk Armijo
    f(x_k + alpha * d_k) <= f(x_k) + c * alpha * gradient()f(x_k).T * dk
    dk: hướng đi
    vd: với GD -> dk = - gradient()*f(x_k)
            newton -> dk = -Hessian_k ^ -1 * gradient()f(x_k)
    '''
    
    def __init__(self, model, loss_fn, direction="gd", lr=1, rho=0.5, c=1e-4, min_alpha=1e-8, max_iter=50, reuse_lr=True):
        super().__init__(lr)
        self.model = model
        self.loss_fn = loss_fn
        self.direction = direction
        self.rho = rho # độ co
        self.c = c 
        self.min_alpha = min_alpha # nếu alpha quá nhỏ cũng dừng line-search
        self.max_iter = max_iter
        self.reuse_lr = reuse_lr # cờ để thuật toán đánh dấu mỗi vòng lặp dùng lr config đầu hay dùng lr tối ưu vòng lặp trước
        self.last_alpha = lr

        # phục vuh linse search của newton
        self.B = None    # ma trận xấp xỉ nghịch đảo Hessian
        self.prev_params = None
        self.prev_grad = None
        
        
    def get_direction(self, grad, params):
        if self.direction == "gd":
            return -grad
        
        elif self.direction == "newton":
            n_params = grad.size
            if self.B is None:
                self.B = np.eye(n_params)

            # d = -B*g
            d = -self.B.dot(grad)

            # nếu có bước trước thì cập nhật BFGS
            if self.prev_params is not None and self.prev_grad is not None:
                s = (params - self.prev_params).reshape(-1, 1)
                y = (grad - self.prev_grad).reshape(-1, 1)
                ys = float(y.T @ s)

                if ys > 1e-12:  # tránh chia 0
                    I = np.eye(n_params)
                    term1 = I - (s @ y.T) / ys
                    term2 = I - (y @ s.T) / ys
                    self.B = term1 @ self.B @ term2 + (s @ s.T) / ys
                else:
                    print("BFGS update skipped (y^T s quá nhỏ)")

            return d
        else:
            raise ValueError("direction must be 'gd' or 'newton'")
            
    
    # --- Backtracking line search ---
    def backtracking(self, d, g):
        f_x = self.loss_fn()
        alpha = self.last_alpha if self.reuse_lr else self.lr
        flat_params = self.model.get_params()

        for _ in range(self.max_iter):
            self.model.set_params(flat_params + alpha * d)
            self.loss_fn.predicts = self.model.forward(self.loss_fn.input)
            f_new = self.loss_fn()

            if f_new <= f_x + self.c * alpha * np.dot(g, d):
                self.last_alpha = alpha
                self.model.set_params(flat_params)
                return alpha

            alpha *= self.rho
            if alpha < self.min_alpha:
                print(f" Alpha quá nhỏ ({alpha:.2e}), dừng line search.")
                break

        self.model.set_params(flat_params)
        return alpha
    
    
    # --- Cập nhật tham số ---
    def update(self):
         # --- 1. Tính loss hiện tại ---
        f_x = self.loss_fn()

        # --- 2. Forward & Backward để có gradient ---
        self.loss_fn.predicts = self.model.forward(self.loss_fn.input)
        grad_out = self.loss_fn.backward()
        self.model.backward(grad_out)

        # --- 3. Lấy gradient & params hiện tại ---
        g = self.model.get_grads().ravel()
        base_params = self.model.get_params().copy()

        # --- 4. Tính hướng đi ---
        d = self.get_direction(g,base_params)

        # --- 5. Backtracking tìm alpha (trả params về base sau khi thử) ---
        alpha = self.backtracking(d, g)

        # --- 6. Update params chính thức ---
        self.model.set_params(base_params + alpha * d)
        
        # 7. Lưu state cho Quasi-Newton
        if self.direction == "newton":
            self.prev_params = base_params.copy()
            self.prev_grad = g.copy()

        return alpha
    
    
    # --- Step cho optimizer ---
    def step(self, model):
        self.update()

            
    
class Momentum(Optimzer):
    
    '''
    class triển khai thuật toán Momentun
    ct thuật toán
    1. cập nhập velocity
        v(t) = momentum * v(t-1) - lr * gradient(t)
    2. cập nhập tham số
        x(t+1) = x(t) + v(t)
    '''
    
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__(lr)
        self.momentum = momentum
        self.velocity = {}
        
    def update(self, param, grad, key=None):
        if key is None:
            raise ValueError("Momentum optimzer requires a unique key.") # truyền để phân biệt là tính velocity cho W hay b
        
        # khởi tạo velocity
        if key not in self.velocity:
            self.velocity[key] = np.zeros_like(grad)
            
        # kiểm tra shape tránh lỗi broadcasting
        if self.velocity[key].shape != grad.shape:
            raise ValueError(f"Shape mismatch for key '{key}': velocity {self.velocity[key].shape} vs grad {grad.shape}")
        
        # cập nhập velocity
        #print("****", grad.shape)
        self.velocity[key] = self.momentum * self.velocity[key] - self.lr * grad
        
        # cập nhập tham số
        return param + self.velocity[key]
    

    def step(self, model):
        self.model = model
        params = model.get_params_not_line_search()
        grads = model.get_grads_not_line_search()

        for (param, key), (grad, _) in zip(params, grads):
            if grad is not None:
                #print(f"{key} grad norm: {np.linalg.norm(grad)}")
                param[...] = self.update(param, grad, key) # param[...] giữ nguyên object nhưng thay đổi toàn bộ giá trị 
            
            
        
    
    
class Adam(Optimzer):
    
    '''
    class triển khai thuật toán adam
    ct thuật toán
    1. cập nhập momentum bậc 1
        m(t) = beta_1 * m(t-1) + (1 - beta_1) * gradient(t)
    2. cập nhập momentum bậc 2
        v(t) = beat_2 * v(t-1) + (1 - beta_1) * gradient(t) ** 2
    3. bias correction
        m_hat(t) = m(t) / (1 - beta_1 ** t)
        v_hat(t) = v(t) / (1 - beta_2 ** t)
    4. cập nhập tham số
        x(t+1) = x(t) - lr * m_hat(t) / (sqrt(v_hat(t) + epsilon))
    '''
    
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        super().__init__(lr)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = {} # momemtum bậc 1
        self.v = {} # momemtum bậc 2
        self.t = {} # thời gian
        
    def update(self, param, grad, key=None):
        if key is None:
            raise ValueError("Adam optimzer requires a unique key.") # truyền để phân biệt là tính velocity cho W hay b
        
        
        # Khởi tạo m, v, t
        if key not in self.m:
            self.m[key] = np.zeros_like(grad)
            self.v[key] = np.zeros_like(grad)
            self.t[key] = 0
            
        self.t[key] += 1
        
        # tính momentum bậc 1
        self.m[key] = self.beta_1 * self.m[key] + (1 - self.beta_1) * grad
        
        # tính momentum bậc 2
        self.v[key] = self.beta_2 * self.v[key] + (1 - self.beta_2) * (grad ** 2)
        
        # bias correction
        m_hat = self.m[key] / (1 - self.beta_1 ** self.t[key])
        v_hat = self.v[key] / (1 - self.beta_2 ** self.t[key])
        
        # cập nhập tham số
        return param - self.lr * (m_hat /(np.sqrt(v_hat) + self.epsilon))
    
    
    def step(self, model):
        self.model = model
        params = model.get_params_not_line_search()
        grads = model.get_grads_not_line_search()

        for (param, key), (grad, _) in zip(params, grads):
            if grad is not None:
                #print(f"{key} grad norm: {np.linalg.norm(grad)}")
                param[...] = self.update(param, grad, key) # param[...] giữ nguyên object nhưng thay đổi toàn bộ giá trị 
            
    

""""
triển khai thuật toán Newton bằng sử dụng Hessian-Vector Product (Hv product) để xấp xỉ ma trận H^-1
để tham khảo có time sẽ check kĩ hơn
class Newton(Optimzer):
    '''
    class triển khai thuật toán newton
    Ct triển khai thuật toán
    x_k+1 = x_k + d
    trong đó d: hướng của đạo hàm
        d = -H^-1 * gradient(x_k) --> công thức newton chuẩn nhưng tính toán rất tốn chi phí và chưa chính xác vì tính toán H^-1
        sử dụng Hessian-Vector Product (Hv product) để xấp xỉ ma trận H^-1
    '''

    def __init__(self, model, loss_fn, lr=0.1, cg_tol=1e-4, cg_maxiter=None, damping=0.05):
        super().__init__(lr)
        self.cg_tol = cg_tol
        self.cg_maxiter = cg_maxiter
        self.model = model
        self.loss_fn = loss_fn
        self.damping = damping

    def gauss_newton_hv(self, v, eps=1e-3):
        flat_params = self.model.get_params()
        y_pred = self.loss_fn.predicts.copy()
        target = self.loss_fn.targets

        # Perturbation để lấy Jv
        self.model.set_params(flat_params + eps * v)
        y_perturbed = self.model.forward(self.loss_fn.input)
        Jv = (y_perturbed - y_pred) / eps

        # Backprop qua loss với predicts bị perturb
        self.loss_fn.predicts = y_pred + Jv
        grad_out = self.loss_fn.backward()
        self.model.backward(grad_out)
        Hv = self.model.get_grads().copy().ravel()

        # Reset state
        self.model.set_params(flat_params)
        self.loss_fn.predicts = y_pred
        self.loss_fn.targets = target
        return Hv
    

    # --- Cập nhật tham số ---
    def update(self, params, grad):
        grad = grad.ravel()

        def Hv_func(v):
            return self.gauss_newton_hv(v)

        # dùng thu viện tính d hướng đạo hàm
        #lin_op = LinearOperator((grads.size, grads.size), matvec=Hv_func)
        #d, info = cg(lin_op, -grads, tol=self.cg_tol, maxiter=self.cg_maxiter)

        #if not is_gpu_enable:
        lin_op = _linearOperator((grad.size, grad.size), matvec=Hv_func)
        d, info = _cg(lin_op, -grad, atol=self.cg_tol, maxiter=self.cg_maxiter)
        #else:
        #    lin_op = _cupy_linearOperator((grad.size, grad.size), matvec=Hv_func)
        #    d, info = _cupy_cg(lin_op, -grad, tol=self.cg_tol, maxiter=self.cg_maxiter)

        if info != 0:
            print(f"CG not converged, info={info}")

        return params + self.lr * self.damping * d
    

    def step(self, model):

        # Forward & Backward để có gradient
        self.loss_fn.predicts = self.model.forward(self.loss_fn.input)
        grad_out = self.loss_fn.backward()
        self.model.backward(grad_out)

        # Lấy params và grads (flatten vector)
        params = self.model.get_params()
        grads = self.model.get_grads()

        # Update params theo Newton step
        new_params = self.update(params, grads)
        model.set_params(new_params)
"""

class QuasiNewton(Optimzer):
    '''
    triển khai thuật toán giả newton Quasi(BFGS)
    thay vì tính ma traanh Hessian ta thay thế H^-1 = B_k
    với B_k bước đầu lấy bằng I 
    sau từng bước lặp cập nhập lại B_k
    '''

    def __init__(self, model, loss_fn, lr=1.0, damping=1e-8):
        super().__init__(lr)
        self.model = model
        self.loss_fn = loss_fn
        self.damping = damping
        self.B = None    # ma trận xấp xỉ nghịch đảo Hessian
        self.prev_params = None
        self.prev_grad = None


    # --- Cập nhật tham số ---
    def update(self, params, grad):
        grad = grad.ravel()

        n_params = grad.size
        if self.B is None:
            # Khởi tạo B = I
            self.B = np.eye(n_params)

        # Hướng đi: d = -B g
        d = -self.B.dot(grad)

        # Cập nhật tham số
        new_params = params + self.lr * d

        # Nếu có thông tin bước trước thì cập nhật BFGS
        if self.prev_params is not None and self.prev_grad is not None:
            s = (new_params - self.prev_params).reshape(-1, 1)
            y = (grad - self.prev_grad).reshape(-1, 1)

            ys = float(y.T @ s)
            if ys > 1e-12:  # tránh chia 0
                I = np.eye(n_params)
                term1 = (I - (s @ y.T) / ys)
                term2 = (I - (y @ s.T) / ys)
                self.B = term1 @ self.B @ term2 + (s @ s.T) / ys
            else:
                print("BFGS update skipped (y^T s quá nhỏ)")

        # Lưu lại cho lần sau
        self.prev_params = new_params.copy()
        self.prev_grad = grad.copy()

        return new_params
    

    def step(self, model):
        # Forward & Backward để có gradient
        self.loss_fn.predicts = self.model.forward(self.loss_fn.input)
        grad_out = self.loss_fn.backward()
        self.model.backward(grad_out)

        # Lấy params và grads
        params = self.model.get_params()
        grads = self.model.get_grads()

        # Update params bằng Quasi-Newton
        new_params = self.update(params, grads)
        model.set_params(new_params)