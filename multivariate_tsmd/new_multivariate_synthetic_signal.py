import numpy as np 
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.interpolate import CubicSpline

##############################################################################################
##############################################################################################
### MOTIF ###
##############################################################################################
##############################################################################################


class Motif(object): 

    def __init__(self,length :int, amplitude :float, motif_fct : callable) -> None:
        """Motif initialization

        Args:
            length (int): Base motif length
            fundamental (float): fundamental frequence for the sum of waveform signal
            motif_fct (callable): function that generates pattern independently of fluctuations
        """
        self.length = length
        self.amplitude = amplitude
        self.motif_fct = motif_fct

    def _occurence_length(self, length_fluctuation = 0.): 
        if length_fluctuation !=0:
            time_offset = (2*np.random.rand(1)-1)*length_fluctuation 
        else: 
            time_offset = 0
        return time_offset
    
    def _occurence_amplitude(self, amplitude_fluctuation = 0.):
        if amplitude_fluctuation!=0: 
            amp = (2*np.random.rand(1)-1)*amplitude_fluctuation
        else: 
            amp = 0
        return amp

    def _time_amplitude(self,length_fluctuation = 0.,amplitude_fluctuation =0.):
        n_time = int((1+self._occurence_length(length_fluctuation))*self.length)
        time = np.linspace(0,1,n_time)
        amp = (1+self._occurence_amplitude(amplitude_fluctuation))*self.amplitude
        return time,amp
    
    def _envelope(self, time):
        """Smooth envelope forcing f(0)=f(1)=0"""
        return np.sin(np.pi * time) ** 2

    
    def get_motif(self, length_fluctuation=0, amplitude_fluctuation=0):
        time, amp = self._time_amplitude(length_fluctuation,amplitude_fluctuation)
        envelope = self._envelope(time)
        return amp * envelope[:, None] * self.motif_fct(time)

class Sin(Motif): 
    def __init__(self, n_actives_dimensions,length, fundamental, amplitude) -> None:
        self.nd = n_actives_dimensions
        self.fundamental = fundamental
        t = np.arange(length)

        self.freq_ = (2*np.pi * fundamental * t / length).reshape(-1, 1).repeat(self.nd, axis=1)
        self.offset_ = 2*np.pi*np.random.rand(length,self.nd)
        self.amp_ = ((2*np.random.rand(length,self.nd)-1)*amplitude)

        def fct(time):
            # time ∈ [0,1] → index continu
            idx = (time * (length - 1)).astype(int)
            return np.sum(
                self.amp_[idx, :, None] *
                np.sin(self.freq_[idx, :, None] * time.reshape(1, 1, -1) + self.offset_[idx, :, None]),
                axis=0
            ).T

        super().__init__(length, amplitude, fct)

class Cubic(Motif):
    def __init__(self, length, fundamental, amplitude) -> None:
        self.fundamental = fundamental
        x = np.linspace(0,1,fundamental+2)
        y = np.hstack((0,np.random.randn(fundamental),0))
        fct = CubicSpline(x,y)
        
        super().__init__(length, amplitude, fct)

##############################################################################################
##############################################################################################
### SIGNAL GENERATOR ###
##############################################################################################
##############################################################################################

class NewMultivariateSignalGenerator(object): 

    def __init__(self,n_motifs:int, n_d=10, n_actives_dimensions_ratio=0.2, motif_length=100, motif_amplitude=1, motif_fundamental =3, motif_type ='Sin',noise_amplitude=0.1,n_novelties=0,length_fluctuation=0.,amplitude_fluctuation=0.,sparsity=0.2,sparsity_fluctuation = 0.,walk_amplitude = 0.,min_rep=2,max_rep=5, n_noninfo_forms: int = 30,
             p_noise_in_noninfo: float = 1, n_motifs_with_common_dimensions: int = 0, n_sub_motifs: int = 0, ratio_common_dimensions: int = 0, n_occurences= None, n_active_dimensions_min=None, n_active_dimensions_max=None) -> None:
        """Signal Generator Initialization

        Args:
            n_motifs (int): number of motifs
            motif_length (int, optional): base pattern length. Defaults to 100.
            motif_amplitude (float, optional): base pattern amplitude. Defaults to 1.
            motif_fundamental (float, optional): pattern fundamental. Defaults to 1.
            motif_type (str, optional): waveform type. Defaults to 'sin'.
            noise_amplitude (float, optional): noise amplitude. Defaults to 0.1.
            n_novelties (int, optional): number of novelties. Defaults to 0.
            length_fluctuation (float, optional): pattern length fluctuation percentage. Defaults to 0..
            amplitude_fluctuation (float, optional): pattern amplitude fluctuation percentage. Defaults to 0..
            sparsity (float, optional): sparsity between pattern. Defaults to 0.2.
            sparsity_fluctuaion (float,optional): random sparsity fluctuation. Defaluts to 0.0
            walk_amplitude (float,optional): random walk amplitude. Defaluts to 0.0
            min_rep (int, optional): minimum motif repetition. Defaults to 2.
            max_rep (int, optional): maximum motif repetition. Defaults to 5.
        """
        self.n_motifs = n_motifs
        self.n_actives_dimensions_ratio=n_actives_dimensions_ratio
        self.n_d =n_d

        self.motif_length = motif_length
        self.motif_amplitude = motif_amplitude
        self.motif_fundamental = motif_fundamental
        self.motif_type = motif_type
        self.noise_amplitude = noise_amplitude
        self.n_novelties = n_novelties
        self.length_fluctuation = length_fluctuation
        self.amplitude_fluctuation = amplitude_fluctuation
        self.sparsity = sparsity
        self.sparsity_fluctuation = sparsity_fluctuation
        self.walk_amplitude = walk_amplitude
        if n_occurences is None:
            self.min_rep = min_rep
            self.max_rep = max_rep
        else: 
            self.min_rep= n_occurences
            self.max_rep= n_occurences
        self.n_noninfo_forms = n_noninfo_forms
        self.p_noise_in_noninfo = p_noise_in_noninfo
        self.n_motifs_with_common_dimensions = n_motifs_with_common_dimensions
        self.n_sub_motifs = n_sub_motifs
        self.ratio_common_dimensions = ratio_common_dimensions
        self.n_active_dimensions_min = n_active_dimensions_min
        self.n_active_dimensions_max = n_active_dimensions_max

        total_motifs = self.n_motifs + self.n_motifs_with_common_dimensions + self.n_novelties

        if n_active_dimensions_min is not None and n_active_dimensions_max is not None:
            self.n_act_d_list = [
                max(1, int(np.random.uniform(n_active_dimensions_min, n_active_dimensions_max) * self.n_d))
                for _ in range(total_motifs)
            ]
        else:
            if isinstance(self.n_actives_dimensions_ratio, (float, int)):
                self.n_actives_dimensions_ratio = [self.n_actives_dimensions_ratio] * total_motifs
            elif isinstance(self.n_actives_dimensions_ratio, list):
                if len(self.n_actives_dimensions_ratio) != total_motifs:
                    raise ValueError(
                        f"La liste n_actives_dimensions_ratio doit avoir la taille {total_motifs} "
                        f"(n_motifs + n_motifs_with_common_dimensions + n_novelties)."
                    )
            else:
                raise TypeError("n_actives_dimensions_ratio doit être un float ou une liste de floats.")
            self.n_act_d_list = [max(1, int(r * self.n_d)) for r in self.n_actives_dimensions_ratio]

            self.n_act_d_list = [max(1, int(r * self.n_d)) for r in self.n_actives_dimensions_ratio]

        n_patterns = self.n_motifs + self.n_novelties

        # Longueurs
        if isinstance(self.motif_length, int):
            self.length_lst_ = (self.motif_length * np.ones(n_patterns)).astype(int)
        else:
            self.length_lst_ = np.random.randint(*self.motif_length, size=n_patterns)

        # Amplitudes
        if isinstance(self.motif_amplitude, int):
            self.amplitude_lst_ = self.motif_amplitude * np.ones(n_patterns)
        else:
            self.amplitude_lst_ = np.random.rand(n_patterns) * (
                self.motif_amplitude[1] - self.motif_amplitude[0]
            ) + self.motif_amplitude[0]


    def _occurence(self): 
        lst = []
        lst = [np.random.randint(self.min_rep,self.max_rep+1) for _ in range(self.n_motifs)]
        lst += [1]*self.n_novelties
        self.occurences_ = np.array(lst, dtype=int)
    
    def _ordering(self):
        arr = []
        for i,occ in enumerate(self.occurences_): 
            arr = np.r_[arr,np.full(occ,i)]
        for i,occ in enumerate(self.com_dim_occurences_,start=1):
            arr = np.r_[arr,np.full(occ,-i)]
        np.random.shuffle(arr)
        self.ordering_ = arr.astype(int)

    def _motifs(self): 
        lst = []
        for i, (m_len, m_amp) in enumerate(zip(self.length_lst_, self.amplitude_lst_)):
            if i < self.n_motifs:
                # Motif normal → ratio défini dans la liste
                n_act_d = self.n_act_d_list[i]
            else:
                # Novelty → ratio aléatoire entre 0 et 1
                rand_ratio = np.random.rand()
                n_act_d = max(1, int(rand_ratio * self.n_d))
            lst.append(globals()[self.motif_type](n_act_d, m_len, self.motif_fundamental, m_amp))
        self.motifs_ = lst

    def _com_dim_occurences(self):
        lst = []
        for _ in range(self.n_motifs_with_common_dimensions):
            for _ in range(self.n_sub_motifs):
                lst.append(np.random.randint(self.min_rep,self.max_rep+1))
        self.com_dim_occurences_ = np.array(lst).astype(int)

    def _create_motifs_with_common_dimensions(self):
        """
        Crée des groupes de motifs composés (multi-dimensionnels) :
        - Chaque groupe contient self.n_sub_motifs sous-motifs
        - Tous les sous-motifs d'un même groupe partagent une forme identique
        sur certaines dimensions communes (ratio défini par ratio_common_dimensions)
        """
        self.common_parts = []
        self.unique_parts = []

        for i in range(self.n_motifs_with_common_dimensions):
            idx_global = self.n_motifs + i
            # Nombre total de dimensions actives pour ce groupe
            n_actives = self.n_act_d_list[idx_global]

            # Nombre de dimensions communes
            n_common = max(1, int(n_actives * self.ratio_common_dimensions))

            # Choisir les dimensions communes
            common_dims = np.random.choice(self.n_d, n_common, replace=False)
            remaining_dims = [d for d in range(self.n_d) if d not in common_dims]

            # Créer une forme commune pour ces dimensions
            base_motif = globals()[self.motif_type](
                len(common_dims),
                self.motif_length,
                self.motif_fundamental,
                self.motif_amplitude,
            )

            # Créer les sous-motifs
            
            for j in range(self.n_sub_motifs):
                n_unique = max(1, n_actives - n_common)
                unique_dims = np.random.choice(
                    remaining_dims,
                    size=min(n_unique, len(remaining_dims)),
                    replace=False
                )
                # motif indépendant pour les dimensions uniques
                motif_obj = globals()[self.motif_type](
                    len(unique_dims),
                    self.motif_length,
                    self.motif_fundamental,
                    self.motif_amplitude,
                )
                self.common_parts.append((base_motif, common_dims))
                self.unique_parts.append((motif_obj, unique_dims))
                # Remplacer les dimensions communes par la forme partagée
                self.active_dimensions.append(common_dims.tolist() + unique_dims.tolist())
            
    def _active_dimensions(self):
        self.active_dimensions = []
        for i in range(self.n_motifs):
            n_act_d = self.n_act_d_list[i]
            self.active_dimensions.append(np.random.choice(self.n_d, size=n_act_d, replace=False))
    

    def generate(self): 
        """
        Asumption: 
        - Max length before first occurence and after last occurence
        - scallability as perecentage of maxlength
        """
        self._occurence()
        self._com_dim_occurences()
        self._ordering()
        self._motifs()
        self._active_dimensions()
        self._create_motifs_with_common_dimensions()
        #number of patterns
        n_patterns = self.n_motifs+self.n_novelties
        #Maximum length
        if isinstance(self.motif_length,int): 
            max_length = self.motif_length
        else: 
            max_length = self.motif_length[0]
        #signal initialisation
        signal_main = [np.zeros((max_length, self.n_d))]
        labels = [np.zeros((max_length,n_patterns+len(self.common_parts)))]
        pos_idx = max_length
        positions = {i: [] for i in np.arange(n_patterns)}
        noninfo_forms = []
        if self.n_noninfo_forms > 0:
            for _ in range(self.n_noninfo_forms):
                wave = globals()[self.motif_type](1, max_length, self.motif_fundamental, self.motif_amplitude)
                form = wave.get_motif(length_fluctuation=0.0, amplitude_fluctuation=0.0).flatten()
                noninfo_forms.append(form)
                
        #signal iteration
        for i,idx in enumerate(self.ordering_): 
            if idx >= 0:
                # -------- Motif "positif" classique --------
                active_pattern = self.motifs_[idx].get_motif(self.length_fluctuation, self.amplitude_fluctuation)
                j = 0
                t_pattern = np.zeros((active_pattern.shape[0], self.n_d))
                for d in range(self.n_d):
                    if d in self.active_dimensions[idx]:
                        t_pattern[:, d] += active_pattern[:, j]
                        j += 1
                    elif np.random.rand() < self.p_noise_in_noninfo or not noninfo_forms: 
                        continue  # on garde les zéros
                    else:
                        form_id = np.random.randint(len(noninfo_forms))
                        form = noninfo_forms[form_id]
                        if len(form) > t_pattern.shape[0]:
                            sample = form[:t_pattern.shape[0]]
                        elif len(form) < t_pattern.shape[0]:
                            pad = t_pattern.shape[0] - len(form)
                            sample = np.concatenate([form, np.zeros(pad)])
                        else:
                            sample = form
                        t_pattern[:, d] += sample
                motif_label = idx  # label positif

            else:
                # -------- Motif avec dimensions communes (négatif) --------
                idx_group = abs(idx) - 1  # correspond à l'indice dans self.common_parts_ etc.
                base_motif, common_dims = self.common_parts[idx_group]
                unique_motif, unique_dims = self.unique_parts[idx_group]

                # on génère les deux parties
                base_pattern = base_motif.get_motif(self.length_fluctuation, self.amplitude_fluctuation)
                unique_pattern = unique_motif.get_motif(self.length_fluctuation, self.amplitude_fluctuation)

                # on assemble dans t_pattern
                T = base_pattern.shape[0]
                t_pattern = np.zeros((T, self.n_d))

                # insérer la forme commune
                for j, d in enumerate(common_dims):
                    t_pattern[:, d] += base_pattern[:, j]

                # insérer la forme unique
                for j, d in enumerate(unique_dims):
                    t_pattern[:, d] += unique_pattern[:, j]

                motif_label = n_patterns + idx_group  # label distinct pour les motifs communs

            # -------- Ajout du motif au signal global --------
            signal_main.append(t_pattern)

            t_label = np.zeros((t_pattern.shape[0], n_patterns + len(self.common_parts)))
            t_label[:, motif_label] = 1
            labels.append(t_label)
            positions[motif_label] = positions.get(motif_label, [])
            positions[motif_label].append((pos_idx, t_pattern.shape[0]))
            pos_idx += t_pattern.shape[0]

    # -------- Ajout du gap --------
            if i < len(self.ordering_) - 1:
                max_sparsity = self.sparsity * max_length
                if max_sparsity > 0:
                    if self.sparsity_fluctuation > 0:
                        length_sparsity = np.random.randint(
                            max(0, int(max_sparsity * (1 - self.sparsity_fluctuation))),
                            int(max_sparsity * (1 + self.sparsity_fluctuation))
                        )
                    else:
                        length_sparsity = int(max_sparsity)
                    signal_main.append(np.zeros((length_sparsity, self.n_d)))
                    labels.append(np.zeros((length_sparsity, n_patterns + len(self.common_parts))))
                    pos_idx += length_sparsity
        #signal ending
        signal_main.append(np.zeros((max_length, self.n_d)))
        labels.append(np.zeros((max_length, n_patterns + len(self.common_parts))))

        #post processing
        sig = np.vstack(signal_main)
        sig += np.random.randn(*sig.shape) * self.noise_amplitude
        sig += np.cumsum(self.walk_amplitude * np.random.randn(*sig.shape), axis=0)

        self.signal_ = sig
        self.labels_ = np.vstack(labels).T
        self.positions_ = positions

        return self.signal_,self.labels_

    def plot(self, color_palette='Plotly'): 
        """Plot du signal avec coloration des motifs et sous-motifs (dimensions communes incluses)."""

        palette = getattr(px.colors.qualitative, color_palette)

        # --- 1️⃣ Prendre en compte les nouveaux motifs ---
        n_patterns = self.n_motifs + self.n_novelties
        n_common = len(getattr(self, "common_parts_", []))
        n_total_patterns = n_patterns + n_common

        n_total_colors = n_total_patterns + 1
        if len(palette) < n_total_colors:
            raise Exception(f"La palette {color_palette} n'a pas assez de couleurs ({len(palette)}) pour {n_total_colors} motifs.")

        # --- 2️⃣ Configuration du signal ---
        signal_length, num_dimensions = self.signal_.shape
        fig = make_subplots(rows=num_dimensions, cols=1, shared_xaxes=True)

        # --- 3️⃣ Tracer le signal brut (gris ou couleur claire) ---
        for dim in range(num_dimensions):
            fig.add_trace(go.Scatter(
                y=self.signal_[:, dim],
                mode='lines',
                marker=dict(color='lightgray'),
                opacity=0.5,
                name=f'dim {dim}',
                showlegend=False
            ), row=dim + 1, col=1)

        # --- 4️⃣ Colorier les motifs (y compris ceux avec dimensions communes) ---
        
        for key, lst in self.positions_.items():
            for i, (start, length) in enumerate(lst):
                time = np.arange(start, start + length)

                # Type de motif selon la clé
                if key < self.n_motifs:
                    name = f"motif {key}"
                    color = palette[key + 1]
                elif key < self.n_motifs + self.n_novelties:
                    name = f"novelty {key - self.n_motifs}"
                    color = palette[key + 1]
                else:
                    name = f"common motif {key - n_patterns}"
                    color = palette[key + 1]

                # --- 5️⃣ Tracer sur les dimensions actives seulement ---
                for dim in range(num_dimensions):
                    #original_dim = self.permutation_[dim]
                    if dim not in self.active_dimensions[key]:  # si ce n'est pas une dimension active
                        continue
                    fig.add_trace(go.Scatter(
                        x=time,
                        y=self.signal_[time, dim],
                        mode='lines',
                        marker=dict(color=palette[key + 1]),
                        name=name,
                        legendgroup=str(key),
                        showlegend=i == 0 and dim == 0
                    ), row=dim + 1, col=1)

        # --- 6️⃣ Ajustement esthétique ---
        fig.update_layout(
            margin=dict(l=10, r=50, t=20, b=10),
            width=1200,
            height=300 * num_dimensions,
        )

        fig.show()
