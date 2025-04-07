# Multi-Agent Reinforcement Learning: Porównanie VDN i QMIX w BenchMARL

## 📌 Opis projektu

Celem projektu jest dostosowanie hiperparametrów i porównanie dwóch popularnych algorytmów wieloagentowego uczenia ze wzmocnieniem — **VDN** (Value Decomposition Networks) i **QMIX** — z wykorzystaniem frameworka **BenchMARL**. Eksperymenty przeprowadzono w środowisku obejmującym **scenariusze kooperacyjne** oraz **kompetytywne**.

Projekt zawiera:
- trening agentów w różnych scenariuszach,
- tuning hiperparametrów (learning rate, discount factor),
- analizę wyników.

---

## 📦 Wykorzystane biblioteki

- `BenchMARL`
- `NumPy`, `Pandas`
- `Matplotlib`
- `PyTorch`

---

## 🔬 Środowiska

### 🟢 Multi-Agent Particle Environment (MPE)
Lekka platforma 2D stworzona do testów MARL.

**Scenariusze:**
- `Simple Spread`: agenci rozkładają się równomiernie.
- `Simple Adversary`: kooperacja i unikanie przeciwnika.
- `Simple Push`: scenariusz rywalizacyjny, w którym agenci konkurują o przesunięcie obiektu na określoną pozycję.

**Cechy:**
- środowisko dyskretne,
- kompatybilne z OpenAI Gym,
- możliwość tworzenia własnych scenariuszy.

---

## 🧠 Opis algorytmów

### ✅ Value Decomposition Networks (VDN)

VDN zakłada, że globalna wartość zespołu agentów \( Q_{\text{tot}} \) może być wyrażona jako suma lokalnych wartości \( Q_i \) dla każdego agenta:

\[
Q_{\text{tot}}(\tau, u) = \sum_{i=1}^{n} Q_i(\tau_i, u_i)
\]

**Cechy:**
- prosta struktura,
- szybki trening,
- ograniczenia przy złożonych zależnościach.

---

### ✅ QMIX

QMIX rozszerza VDN przez zastosowanie **nieliniowej, monotonicznej funkcji mieszającej**:

\[
\frac{\partial Q_{\text{tot}}}{\partial Q_i} \geq 0
\]

**Cechy:**
- lepsza reprezentacja złożonych zależności,
- centralizowane trenowanie z zdecentralizowanym wykonaniem,
- wsparcie dla dynamicznych środowisk.

---

## ⚙️ Plan eksperymentów

- Porównanie QMIX i VDN na zadaniach z MPE.
- Testy kooperacyjne: `Simple Spread`.
- Testy kompetytywne: `Simple Push`,  `Simple Adversary`.
- Różne konfiguracje hiperparametrów.

## ⚗️ Eksperymenty z hiperparametrami

W celu znalezienia najlepszej konfiguracji dla algorytmów VDN i QMIX, przeprowadzono serię eksperymentów z różnymi wartościami hiperparametrów. Punktem wyjścia była domyślna konfiguracja BenchMARL (przykład poniżej):

default_config = {
    'sampling_device': 'cuda',
    'train_device': 'cuda',
    'buffer_device': 'cuda',
    'share_policy_params': True,
    'prefer_continuous_actions': False,
    'collect_with_grad': False,
    'parallel_collection': False,
    'gamma': 0.99,
    'lr': 0.01,
    'adam_eps': 1.0e-8,
    'clip_grad_norm': 0.5,
    'clip_grad_val': None,
    'soft_target_update': True,
    'polyak_tau': 0.005,
    'hard_target_update_frequency': 100,
    'exploration_eps_init': 1.0,
    'exploration_eps_end': 0.05,
    'exploration_anneal_frames': 1000000,
    'max_n_frames': 1000000,
    'on_policy_collected_frames_per_batch': 2048,
    'on_policy_n_envs_per_worker': 1,
    'on_policy_n_minibatch_iters': 4,
    'on_policy_minibatch_size': 64,
    'off_policy_collected_frames_per_batch': 100,
    'off_policy_n_envs_per_worker': 1,
    'off_policy_n_optimizer_steps': 100,
    'off_policy_train_batch_size': 512,
    'off_policy_memory_size': 1000000,
    'off_policy_init_random_frames': 50000,
    'off_policy_use_prioritized_replay_buffer': False,
    'off_policy_prb_alpha': 0.6,
    'off_policy_prb_beta': 0.4,
    'evaluation': True,
    'render': False,
    'evaluation_interval': 10000,
    'evaluation_episodes': 10,
    'evaluation_deterministic_actions': True,
    'project_name': 'benchmarl',
    'create_json': True,
    'save_folder': 'results',
    'restore_file': None,
    'restore_map_location': None,
    'checkpoint_interval': 100,
    'checkpoint_at_end': True,
    'keep_checkpoints_num': 1,
    'max_n_iters': 5000,
    'loggers': [],
}


---

## 🧩 Wpływ hiperparametrów na algorytmy MARL

### 1. `lr` (Learning Rate)
- **Opis:** Określa szybkość aktualizacji wag modelu podczas trenowania.
- **Wpływ:** Zbyt wysoka wartość może prowadzić do niestabilności modelu, co objawia się dużymi wahania w nagrodach. Zbyt niska wartość może spowodować bardzo wolne uczenie się i zbyt małe zmiany w polityce agenta.

### 2. `clip_grad_norm`
- **Opis:** Ogranicza długość gradientu.
- **Wpływ:** Zapobiega eksplozji gradientów, co może być szczególnie problematyczne w złożonych środowiskach. Wyłączenie tego parametru (`None`) może prowadzić do niestabilnych wyników.

### 3. `polyak_tau`
- **Opis:** Współczynnik miękkiego aktualizowania wag targetowych w sieci neuronowej.
- **Wpływ:** Wyższe wartości przyspieszają aktualizację wag, co może skutkować szybszą konwergencją, ale również większymi fluktuacjami w wynikach. Zbyt niskie wartości mogą spowolnić adaptację do nowych danych.

### 4. `off_policy_memory_size`
- **Opis:** Rozmiar bufora pamięci do przechowywania doświadczeń agenta.
- **Wpływ:** Większy bufor umożliwia lepszą dywersyfikację danych treningowych, co może poprawić jakość polityki. Zbyt duży bufor może jednak spowodować problemy z pamięcią i czasem przetwarzania.

### 5. `off_policy_train_batch_size`
- **Opis:** Liczba próbek pobieranych z bufora do jednego kroku optymalizacji.
- **Wpływ:** Mniejsze wartości mogą powodować większy szum w gradientach, co utrudnia konwergencję. Z drugiej strony, zbyt duże wartości mogą spowodować dłuższy czas treningu i zwiększone obciążenie pamięci.

### 6. `prefer_continuous_actions`
- **Opis:** Wskazuje, czy model powinien obsługiwać akcje ciągłe.
- **Wpływ:** Włączenie tej opcji wpływa na kompatybilność z różnymi środowiskami. W przypadku środowisk z dyskretnymi akcjami może to prowadzić do nieoptymalnych strategii.

### 7. `exploration_eps_init`
- **Opis:** Początkowa wartość parametru eksploracji w strategii ε-greedy.
- **Wpływ:** Im wyższa wartość, tym bardziej eksploracyjne działania na początku treningu, co może pomóc w odkrywaniu lepszych strategii. Zbyt niski parametr może prowadzić do lokalnych minimów.

### 8. `exploration_anneal_frames`
- **Opis:** Liczba kroków, po których wartość ε jest zmniejszana do wartości końcowej (`exploration_eps_end`).
- **Wpływ:** Krótszy okres prowadzi do szybszego przejścia z eksploracji do eksploatacji, co może być korzystne w prostych środowiskach, ale w bardziej złożonych może prowadzić do utraty szans na odkrycie lepszych strategii.

---

## 📝 Wnioski

- **Stabilność algorytmów:** VDN w większości przypadków wykazuje większy rozrzut wyników (szersze przedziały wartości), podczas gdy QMIX jest bardziej stabilny, co przejawia się w węższych przedziałach wartości. Oznacza to, że QMIX może lepiej radzić sobie w bardziej złożonych scenariuszach.

- **Czas treningu:** Średni czas potrzebny na uczenie algorytmu QMIX był o 30% większy niż w przypadku VDN. Oznacza to, że QMIX może wymagać większych zasobów obliczeniowych, co warto uwzględnić przy wyborze algorytmu.

- **Wrażliwość na hiperparametry:** Oba algorytmy wykazały znaczną wrażliwość na wartość współczynnika uczenia oraz rozmiar partii treningowej. Odpowiednie dostrojenie tych parametrów jest kluczowe dla osiągnięcia wysokiej jakości wyników.

- **Eksploracja:** Zwiększenie zakresu eksploracji przyczyniło się do lepszego odkrywania strategii, ale jednocześnie wprowadzało większą zmienność wyników. Sugeruje to, że istnieje optymalny poziom eksploracji, który powinien być ustalany w zależności od specyfiki zadania.

- **Optymalizacja architektury:** W przeprowadzonych eksperymentach skupiono się na optymalizacji hiperparametrów, nie ingerując w rozmiary architektury sieci neuronowych. To podejście pozwala na maksymalne wykorzystanie potencjału modelu, co jest kluczowe dla efektywności przy niższych kosztach obliczeniowych. Mniejsze architektury zapewniają szybsze trenowanie oraz mniejsze zużycie zasobów.


