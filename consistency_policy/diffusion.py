import torch
import numpy
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.distributions import Beta
import random
import math
from einops import rearrange, repeat


from consistency_policy.utils import append_dims, reduce_dims

class Karras_Scheduler():
    def __init__(
            self,
            time_min,
            time_max,
            rho,
            bins,
            solver,
            time_sampler,
            scaling,
            data_std,
            
            #log normal sampling
            P_std = 1.2,
            P_mean = -1.2,

            weighting = "none",

            #euler marayuma corrector step
            alpha = -1, #ratio btwn corr step and ode step
            friction = -1, #friction for corr step

            #time chunking
            beta = .0,

            name = "singular",
            **kwargs, #for compatibility with old code, should clean up later
    ):
        
        self.time_min = time_min
        self.time_max = time_max
        self.rho = rho
        self.bins = bins
        self.data_std = data_std

        self.solver = solver
        self.time_sampler = time_sampler
        self.scaling = scaling

        self.weighting = weighting

        #log normal sampling
        self.P_std = P_std
        self.P_mean = P_mean

        #langevin corrector step
        self.alpha = alpha
        self.friction = friction


        #time chunking
        self.beta = beta
        self.bins_min = 0
        self.bins_max = bins

        self.name = name
        print("Using scheduler {}".format(self.name))

        if "corr" in self.solver:
            if self.alpha < 0:
                raise ValueError("alpha must be specified for corr solver")
            if self.friction < 0:
                raise ValueError("friction must be specified for corr solver")

    # ==================== MAIN ====================
    def step(self, model, samples, t, next_t, clamp=False):
        if self.solver == 'euler' or self.solver == 'first_order':
            return self.euler_solver(model, samples, t, next_t, clamp = clamp)
        elif self.solver == 'heun' or self.solver == 'second_order':
            return self.heun_solver(model, samples, t, next_t, clamp = clamp)
        elif self.solver == 'third':
            return self.third_order_solver(model, samples, t, next_t, clamp = clamp)
        elif self.solver == 'fourth':
            return self.fourth_order_solver(model, samples, t, next_t, clamp = clamp)
        elif self.solver == 'second_order_corr':
            return self.second_order_corr_solver(model, samples, t, next_t, clamp = clamp)
        else:
            raise ValueError(f"Unknown solver {self.solver}")    

    def calc_out(self, model, trajectory: torch.Tensor, times: torch.Tensor, clamp=False):
        if self.scaling == "boundary":
            c_skip, c_out, c_in = [append_dims(c, trajectory.ndim) for c in self.get_scalings_for_boundary_condition(times)]
        elif self.scaling == "no_boundary":
            c_skip, c_out, c_in = [append_dims(c, trajectory.ndim) for c in self.get_scalings(times)]
        else:
            raise ValueError(f"Unknown scaling {self.scaling}")

        if times.ndim > 1:
            times = reduce_dims(times, 1)
               
        rescaled_times = 1000 * 0.25 * torch.log(times + 1e-44) # *1000 to make it more stable
        model_output = model(trajectory * c_in, rescaled_times)

        out = model_output * c_out + trajectory * c_skip
        if clamp:
            out = out.clamp(-1.0, 1.0) #this should only happen at inference time

        return out
    
    def add_noise(self, trajectory: torch.Tensor, times: torch.Tensor):
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        return trajectory + self.trajectory_time_product(noise, times)
    
    def sample_inital_position(self, trajectory, generator):

        traj = torch.randn(
            size= trajectory.shape, 
            dtype=trajectory.dtype,
            device=trajectory.device,
            generator=generator)
        
        return traj ## * self.time_max ## Reducing Initial Variance by not multiplying by time_max
    
    # ==================== TIME SAMPLERS ====================
 
    def sample_times(self, trajectory: torch.Tensor, time_sampler = None):
        time_sampler = time_sampler if time_sampler is not None else self.time_sampler
        batch = trajectory.shape[0]
        device = trajectory.device

        if time_sampler == "uniform":
            return self.uniform_sampler(batch, device)
        elif time_sampler == "log_normal":
            return self.log_normal_sampler(batch, device)
        elif time_sampler == "uniform_time_chunked":
            return self.uniform_time_chunked_sampler(batch, device)
        elif time_sampler == "ctm_dsm":
            return torch.cat((self.ctm_dsm_sampler(int(math.ceil(batch/2)), device)[0], self.log_normal_sampler(int(math.floor(batch/2)), device)[0]), dim = 0), None
        else:
            raise ValueError(f"Unknown sampler {time_sampler}")
    
    def uniform_sampler(self, batch, device):
        timesteps = torch.randint(
            0,
            self.bins - 1,
            (batch,),
            device=device,
        ).long()

        return self.timesteps_to_times(timesteps), self.timesteps_to_times(timesteps+1)

    def uniform_time_chunked_sampler(self, batch, device):

        #these should all cross the boundary
        if random.random() < self.beta:
            #timesteps should just all be bins_max
            timesteps = torch.ones(
                (int(batch), ),
                device=device,
            ).long() * self.bins_max

            times, next_times = self.timesteps_to_times(timesteps), self.timesteps_to_times(timesteps-1)
            return times, next_times
        
        #these should all be inside the current chunk
        timesteps = torch.randint(
            self.bins_min,
            self.bins_max - 1,
            (batch,),
            device=device,
        ).long()

        return self.timesteps_to_times(timesteps), self.timesteps_to_times(timesteps+1)
       
    def get_boundary(self):
        """
        Returns the lower time boundary of the current chunk, which corresponds to the max bin
        """
        return self.timesteps_to_times(torch.tensor(self.bins_max))

    def log_normal_sampler(self, batch, device):
        """
            Sample times such that mean and variance of the ln of the times is -1.2, 1.2.
            Goal is to target training towards the beginning of the diffusion process
        """

        sigma = (torch.randn((batch,), device=device) * self.P_std + self.P_mean).exp()

        return sigma, sigma

    def ctm_dsm_sampler(self, batch, device):
        sigma_max = self.time_max
        sigma_min = self.time_min
        ro = self.rho

        # Generate random samples uniformly from the interval [0, 0.7]
        xi_samples = torch.rand((batch,), device=device) * 0.7
        # Apply the transformation to these samples
        transformed_samples = (sigma_max**(1/ro) + xi_samples*(sigma_min**(1/ro) - sigma_max**(1/ro)))**ro
        transformed_samples = transformed_samples.clamp(sigma_min, sigma_max)
        return transformed_samples, transformed_samples

    def timesteps_to_times(self, timesteps: torch.LongTensor):
        t = self.time_max ** (1 / self.rho) + timesteps / (self.bins - 1) * (
            self.time_min ** (1 / self.rho) - self.time_max ** (1 / self.rho)
        )

        t = t**self.rho

        return t.clamp(self.time_min, self.time_max)
    
    def times_to_timesteps(self, times: torch.Tensor):
        r = 1 / self.rho
        timesteps = (times ** r - self.time_max ** r) * (self.bins - 1) / (self.time_min ** r - self.time_max ** r)

        timesteps = torch.round(timesteps)
        return timesteps.long()


    # ==================== WEIGHTINGS ====================
    def get_weights(self, times, next_times, weighting = None):
        """
        Returns weights to scale loss by.
        Currently supports ICT and Karras weighting
        """
        weighting = weighting if weighting is not None else self.weighting

        if weighting == "none":
            return None
        elif weighting == "ict":
            return self.get_ict_weightings(times, next_times)
        elif weighting == "karras":
            return self.get_karras_weightings(times)
        else:
            raise ValueError(f"Unknown weighting {weighting}")

    def get_ict_weightings(self, times, next_times):
        return 1 / (times - next_times)

    def get_karras_weightings(self, times, **kwargs):
        return (times**2 + self.data_std**2)/((times * self.data_std)**2)
    
    # ==================== PARAMETIRIZATIONS ====================
    def get_scalings(self, time):
        c_skip = self.data_std**2 / (time**2 + self.data_std**2)
        c_out = time * self.data_std / ((time**2 + self.data_std**2) ** 0.5)
        c_in = 1 / (time**2 + self.data_std**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, time):
        c_skip = self.data_std**2 / (
            (time - self.time_min) ** 2 + self.data_std**2
        )
        c_out = (
            (time - self.time_min)
            * self.data_std
            / (time**2 + self.data_std**2) ** 0.5
        )
        c_in = 1 / (time**2 + self.data_std**2) ** 0.5
        return c_skip, c_out, c_in


    # ==================== SOLVERS ====================
    @torch.no_grad()
    def euler_solver(self, model, samples, t, next_t, clamp = False):
        dims = samples.ndim
        y = samples
        step = append_dims((next_t - t), dims)
        
        denoisedy = self.calc_out(model, y, t, clamp = clamp)
        dy = (y - denoisedy) / append_dims(t, dims)

        y_next = samples + step * dy 

        return y_next
    
    @torch.no_grad()
    def heun_solver(self, model, samples, t, next_t, clamp = False):
        dims = samples.ndim
        y = samples
        step = append_dims((next_t - t), dims)
        
        denoisedy = self.calc_out(model, y, t, clamp = clamp)
        dy = (y - denoisedy) / append_dims(t, dims)


        y_next = samples + step * dy 

        denoisedy_next = self.calc_out(model, y_next, next_t, clamp = clamp)
        dy_next = (y_next - denoisedy_next) / append_dims(next_t, dims)

        y_next = samples + step * (dy + dy_next) / 2



        return y_next

    @torch.no_grad()
    def third_order_solver(self, model, samples, t, next_t, clamp = False):
        dims = samples.ndim
        y = samples
        step = next_t - t
        
        denoisedy = self.calc_out(model, y, t, clamp = clamp)
        dy = (y - denoisedy) / append_dims(t, dims)

        y_next = samples + append_dims(step, dims) * dy 

        denoisedy_next = self.calc_out(model, y_next, next_t, clamp = clamp)
        dy_next = (y_next - denoisedy_next) / append_dims(next_t, dims)

        y_mid = samples + append_dims(step, dims) / 2 * (dy + dy_next) / 2

        denoisedy_mid = self.calc_out(model, y_mid, t + step/2, clamp = clamp)
        dy_mid = (y_mid - denoisedy_mid) / append_dims(t + step/2, dims)

        dy_final = (dy + 4 * dy_mid + dy_next) / 6
        y_third = samples + append_dims(step, dims) * dy_final

        return y_third

    @torch.no_grad()
    def fourth_order_solver(self, model, samples, t, next_t, clamp = False):
        dims = samples.ndim
        y = samples
        step = next_t - t
        
        pred = self.calc_out(model, y, t, clamp = clamp)
        k_1 = (y - pred) / append_dims(t, dims)

        y_2 = samples + append_dims(step, dims) / 2 * k_1
        pred = self.calc_out(model, y_2, t + step/2, clamp = clamp)
        k_2 = (y_2 - pred) / append_dims(t + step/2, dims)

        y_3 = samples + append_dims(step, dims) / 2 * k_2
        pred = self.calc_out(model, y_3, t + step/2, clamp = clamp)
        k_3 = (y_3 - pred) / append_dims(t + step/2, dims)

        y_4 = samples + append_dims(step, dims) * k_3
        pred = self.calc_out(model, y_4, t + step, clamp = clamp)
        k_4 = (y_4 - pred) / append_dims(t + step, dims)

        y_next = samples + append_dims(step, dims) / 6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)

        return y_next

    @torch.no_grad()
    def second_order_corr_solver(self, model, samples, t, next_t, clamp = False):
        dims = samples.ndim
        x_0 = self.heun_solver(model, samples, t, t + (next_t - t) * (1-self.alpha), clamp = clamp)
        t_0 = t + (next_t - t) * (1-self.alpha)
        step = append_dims((next_t - t_0), dims)
        #sample v_0 from guassian noise
        v_0 = torch.randn_like(samples)
        #sample brownian noise
        w_t = torch.randn_like(samples)

        #integrate velocity
        drift = self.calc_out(model, x_0, t_0, clamp = clamp) - v_0 * self.friction
        diffusion = w_t  * (2*self.friction)**0.5
        v_t = v_0 + drift * step + diffusion * step.abs().sqrt() #abs because step is negative, but we don't have to change the sign of + diffusion b/c diffusion is a guassian

        #integrate position
        x_t = x_0 + (v_0 + v_t) / 2 * step


        return x_t

    # ==================== HELPERS ====================
    @staticmethod
    def trajectory_time_product(traj: torch.Tensor, times: torch.Tensor):
        return torch.einsum("b T D, b -> b T D ", traj, times)
    
class PFGMPP_Scheduler(Karras_Scheduler):
    def __init__(self,
            time_min,
            time_max,
            rho,
            bins,
            solver,
            time_sampler,
            scaling,
            data_std,
            
            #log normal sampling
            P_std = 1.2,
            P_mean = -1.2,

            weighting = "none",

            #euler marayuma corrector step
            alpha = -1, #ratio btwn corr step and ode step
            friction = -1, #friction for corr step

            #pfgm++
            D = -1,
            N = -1,

            #time chunking
            beta = .0,

            name = "singular",
            **kwargs, #for compatibility with old code, should clean up later):
    ):
        super().__init__(
            time_min = time_min,
            time_max = time_max,
            rho = rho,
            bins = bins,
            solver = solver,
            time_sampler = time_sampler,
            scaling = scaling,

            data_std = data_std,
            P_std = P_std,
            P_mean = P_mean,
            weighting = weighting,
            alpha = alpha,
            friction = friction,
            beta = beta,
            name = name,

        )
        self.D = D
        if self.D == -1:
            raise ValueError("D must be specified for pfgmpp")

        self.N = N
        if self.N == -1:
            raise ValueError("N must be specified for pfgmpp")
            
        print("Using pfgmpp with D = {}, N = {}".format(self.D, self.N))

    def sample_inital_position(self, trajectory, generator):
        def rand_beta_prime(size, device, N, D):
                # sample from beta_prime (N/2, D/2)
                beta_gen = Beta(torch.FloatTensor([N / 2.]), torch.FloatTensor([D / 2.]))

                sample_norm = beta_gen.sample().to(device).double()
                # inverse beta distribution
                inverse_beta = sample_norm / (1-sample_norm)


                sample_norm = torch.sqrt(inverse_beta) * self.time_max * np.sqrt(D)
                gaussian = torch.randn(size[0], N).to(sample_norm.device)
                unit_gaussian = gaussian / torch.norm(gaussian, p=2)
                traj = unit_gaussian * sample_norm

                return traj.view(size)

        if self.N != trajectory.shape[-1] * trajectory.shape[-2]:
            raise ValueError("N must be equal to T * D for pfgmpp but N is {} and T * D is {}".format(self.N, trajectory.shape[-1] * trajectory.shape[-2]))

        traj = rand_beta_prime(trajectory.shape, trajectory.device,
                N=self.N,
                D=self.D,
            )
            
        return traj
    
    def add_noise(self, trajectory: torch.Tensor, times: torch.Tensor):
        if self.N != trajectory.shape[-1] * trajectory.shape[-2]:
            raise ValueError("N must be equal to T * D for pfgmpp but N is {} and T * D is {}".format(self.N, trajectory.shape[-1] * trajectory.shape[-2]))

        r = times.double() * np.sqrt(self.D).astype(np.float64)

        # Sampling from inverse-beta distribution
        samples_norm = np.random.beta(a=self.N / 2., b=self.D / 2.,
                                        size=trajectory.shape[0]).astype(np.double)

        samples_norm = np.clip(samples_norm, 1e-3, 1-1e-3)

        inverse_beta = samples_norm / (1 - samples_norm + 1e-8)
        inverse_beta = torch.from_numpy(inverse_beta).to(trajectory.device).double()
        # Sampling from p_r(R) by change-of-variable
        samples_norm = r * torch.sqrt(inverse_beta + 1e-8)
        samples_norm = samples_norm.view(len(samples_norm), -1)
        # Uniformly sample the angle direction
        gaussian = torch.randn(trajectory.shape[0], self.N).to(samples_norm.device)
        unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)
        # Construct the perturbation for x
        perturbation_x = unit_gaussian * samples_norm
        perturbation_x = perturbation_x.float()

        n = perturbation_x.view_as(trajectory) # B X (T X D) -> B X T X D
        return trajectory + n

class CTM_Scheduler(Karras_Scheduler):
    def __init__(self,
            time_min,
            time_max,
            rho,
            bins,
            solver,
            time_sampler,
            scaling,
            data_std,
            
            #log normal sampling
            P_std = 1.2,
            P_mean = -1.2,

            weighting = "none",

            #euler marayuma corrector step
            alpha = -1, #ratio btwn corr step and ode step
            friction = -1, #friction for corr step

            #time chunking
            beta = .0,

            #ctm
            ode_steps_max = -1,

            name = "singular",
            **kwargs, #for compatibility with old code, should clean up later):
    ):

        if ode_steps_max == -1:
            raise ValueError("ode_steps_max must be specified for CTM scheduler")

        self.ode_steps_max = ode_steps_max

        super().__init__(
            time_min = time_min,
            time_max = time_max,
            rho = rho,
            bins = bins,
            solver = solver,
            time_sampler = time_sampler,
            scaling = scaling,

            data_std = data_std,
            P_std = P_std,
            P_mean = P_mean,
            weighting = weighting,
            alpha = alpha,
            friction = friction,
            beta = beta,
            name = name,

        )
        print("Using CTM scheduler")
   

    def CTM_calc_out(self, model, trajectory: torch.Tensor, times: torch.Tensor, stops: torch.Tensor, clamp=False):     
        if self.scaling == "boundary":
            c_skip, c_out, c_in = [append_dims(c, trajectory.ndim) for c in self.get_scalings_for_boundary_condition(times)]
        elif self.scaling == "no_boundary":
            c_skip, c_out, c_in = [append_dims(c, trajectory.ndim) for c in self.get_scalings(times)]
        else:
            raise ValueError(f"Unknown scaling {self.scaling}")

        if times.ndim > 1:
            times = reduce_dims(times, 1)

        if stops.ndim > 1:
            stops = reduce_dims(stops, 1)

               
        rescaled_times = (1000 * 0.25 * torch.log(times + 1e-44)).expand(trajectory.shape[0]) # *1000 to make it more stable
        rescaled_stops = (1000 * 0.25 * torch.log(stops + 1e-44)).expand(trajectory.shape[0])

        model_output = model(trajectory * c_in, rescaled_times, rescaled_stops)

        out = model_output * c_out + trajectory * c_skip #g_theta

        ratio = (stops/times).unsqueeze(-1).unsqueeze(-1).expand(*out.shape)
    
        out = trajectory * ratio + out * (1-ratio) #G_theta 

        if clamp:
            out = out.clamp(-1.0, 1.0) #this should only happen at inference time

        return out

    def sample_times(self, trajectory: torch.Tensor, time_sampler = None):
        time_sampler = time_sampler if time_sampler is not None else self.time_sampler
        batch = trajectory.shape[0]
        device = trajectory.device

        #this sampler returns t, s, u as bins
        if time_sampler == "ctm":
            #t is uniform over bins
            t = torch.randint(
                0,
                self.bins,
                (batch,),
                device=device,
            ).long()
            #s is uniform over bins greater than t
            s = torch.cat([torch.randint(int(t_i.item()), self.bins+1, (1,)) for t_i in t]).to(device)

            #u is uniform over bins between t and s
            u = torch.cat([

                    (torch.randint(int(t_i.item()), int((s_i+1).item()), (1,)))
                    for t_i, s_i in zip(t, s) #might want to swap bound vs clamp (s_i vs self.ode_max) depending on their relative distributions
                
                ]).to(device)

            maxes = t + self.ode_steps_max
            mask = (u > maxes).float()
            u = u * (1 - mask) + maxes * mask


            return t, s, u
        
        if time_sampler == "ctm_to_cm_ln":
            #t is log normal with mean -1.2, std 1.2
            t, _ = self.log_normal_sampler(batch, device)
            t = self.times_to_timesteps(t)
            #s is min
            s = torch.tensor([self.time_min], device=device).expand(batch).long()
            #u is 1 less than t
            u = t + 1

            return t, s, u
        
        if time_sampler == "ctm_to_cm":
            #t is uniform over bins
            t = torch.randint(
                0,
                self.bins,
                (batch,),
                device=device,
            ).long()
            #s is min
            s = torch.tensor([self.time_min], device=device).expand(batch).long()
            #u is 1 less than t
            u = t + 1

            return t, s, u

        else:
            return super().sample_times(trajectory, time_sampler = time_sampler)

    #CTM needs to handle step size 0
    @torch.no_grad()
    def heun_solver(self, model, samples, t, next_t, clamp = False):
        dims = samples.ndim
        y = samples
        step = append_dims((next_t - t), dims)
        mask = (step == 0).float()
        
        denoisedy = self.calc_out(model, y, t, clamp = clamp)
        dy = (y - denoisedy) / (append_dims(t, dims) + mask)


        y_next = samples + step * dy 

        denoisedy_next = self.calc_out(model, y_next, next_t, clamp = clamp)
        dy_next = (y_next - denoisedy_next) / (append_dims(next_t, dims) + mask)

        y_next = samples + step * (dy + dy_next) / 2

        y_next = y_next * (1 - mask) + samples * mask


        return y_next



def params_to_scheduler(params, i):
    try: 
        time_min = params["time_min"][i]
        time_max = params["time_max"][i]
        rho = params["rho"][i]
        bins = params["bins"][i]
        solver = params["solver"][i]
        time_sampler = params["time_sampler"][i]
        scaling = params["scaling"][i]
        use_c_in = params["use_c_in"][i]
        data_std = params["data_std"][i]
        friction = params["friction"][i]
        alpha = params["alpha"][i]
        name = params["name"][i]
    except:
        raise ValueError("Could not find all params for scheduler. Make sure all params are specified in the config file")

    new_scheduler = Karras_Scheduler(
        time_min = time_min,
        time_max = time_max,
        rho = rho,
        bins = bins,
        solver = solver,
        time_sampler = time_sampler,
        scaling = scaling,
        use_c_in = use_c_in,
        data_std = data_std,
        friction = friction,
        alpha = alpha,
        name = name,
    )

    return new_scheduler    


def Huber_Loss(pred, target, delta = 0, weights = None):
    """
    Computes psuedo-huber loss of pred and target of shape (batch, time, dim)
    
    Delta is the boundary between l_1 and l_2 loss. At delta = 0, this is just MSE loss.
    Setting delta = -1 calculates iCT's recommended delta given data size.
    
    Also supports weighting of loss
    """

    if delta == -1:
        delta = math.sqrt(math.prod(pred.shape[1:])) * .00054

    mse = F.mse_loss(pred, target, reduction = "none")

    loss = torch.sqrt(mse**2 + delta**2) - delta

    if weights is not None:
        loss = torch.einsum("b T D, b -> b T D", loss, weights)



    return loss.mean()
