# Micro-Publication Review

Date: 2026-04-01

## Bottom line

The strongest publication-grade insight currently supported by this repository is:

**A minimal biological NCAP controller can switch locomotor timing and amplitude from direct substrate cues, but the current MuJoCo implementation does not yet test true anisotropic-friction-driven crawling.**

That is a useful result because it separates:

1. cue-driven gait modulation,
2. scalar viscosity/friction changes, and
3. genuine mechanically grounded crawl emergence from anisotropic drag/contact.

This distinction is important for any comparison with PyElastica or with biological claims about crawl versus swim gait generation.

## What the repository actually implements

### 1. The controller already receives explicit environment labels

In the progressive environment, observations include:

- `fluid_viscosity`
- `environment_type`
- `in_water_zone`
- `in_land_zone`

See [progressive_mixed_env.py](/data/work/nma_nai_on/swimmer/environments/progressive_mixed_env.py#L691) through [progressive_mixed_env.py](/data/work/nma_nai_on/swimmer/environments/progressive_mixed_env.py#L741).

The curriculum trainer then reconstructs a compact environment vector and passes it directly into the model during training. See [curriculum_trainer.py](/data/work/nma_nai_on/swimmer/training/curriculum_trainer.py#L310) through [curriculum_trainer.py](/data/work/nma_nai_on/swimmer/training/curriculum_trainer.py#L333).

### 2. The biological NCAP uses that cue to change gait parameters

The biological NCAP explicitly changes:

- torque amplitude from viscosity,
- oscillator period from land vs water,
- proprioceptive gain from the same environment signal.

See [biological_ncap.py](/data/work/nma_nai_on/swimmer/models/biological_ncap.py#L167) through [biological_ncap.py](/data/work/nma_nai_on/swimmer/models/biological_ncap.py#L233).

The enhanced model does the same in a stronger form by switching water and land frequency/amplitude scales. See [enhanced_biological_ncap.py](/data/work/nma_nai_on/swimmer/models/enhanced_biological_ncap.py#L313) through [enhanced_biological_ncap.py](/data/work/nma_nai_on/swimmer/models/enhanced_biological_ncap.py#L331).

### 3. The environment changes scalar viscosity and isotropic contact coefficients

The older mixed environment changes:

- global viscosity,
- MuJoCo geom friction triplets such as `[0.1, 0.005, 0.0001]` in water and `[0.2, 0.05, 0.01]` on land.

See [mixed_environment.py](/data/work/nma_nai_on/swimmer/environments/mixed_environment.py#L123) through [mixed_environment.py](/data/work/nma_nai_on/swimmer/environments/mixed_environment.py#L145).

This is not segment-wise anisotropic friction aligned to the worm body. It is a medium switch plus standard MuJoCo contact parameters.

## What that means scientifically

### Supported claim

The repository supports the claim that:

**Direct environmental sensing is sufficient to let a compact biological controller modulate locomotor rhythm across media.**

This is already aligned with the project’s internal results that direct environmental adaptation outperforms LSTM-style memory.

### Unsupported claim

The repository does **not** yet support the stronger claim that:

**Anisotropic friction itself is responsible for the observed crawl gait change in the current MuJoCo experiments.**

Why:

- the controller gets privileged medium labels,
- the environment mostly changes scalar viscosity,
- the contact model is not currently aligned with local body orientation,
- there is no ablation showing behavior when explicit environment cues are removed.

So if the gait changes now, the clean interpretation is:

**the gait is cue-conditioned and medium-conditioned, not yet proven to be anisotropic-friction-driven.**

## Literature check

### 1. Crawling in C. elegans is strongly linked to anisotropic drag

Shen et al. 2012 estimated that during crawling on wet agar the normal and tangential surface drag coefficients are about 222 and 22, with a ratio near 10, and argued that gait can be controlled by changing the surface drag coefficients.

Source:
- https://pubmed.ncbi.nlm.nih.gov/22735527/

This is directly relevant to the collaborator’s point. If you want biologically grounded crawling, directional drag anisotropy is not a side detail.

### 2. Proprioception and sensory feedback are central to locomotor adaptation

Jia et al. 2023 showed that C. elegans adapts locomotor amplitude through a proprioceptive feedback circuit involving dopamine and neuropeptide signaling.

Source:
- https://pmc.ncbi.nlm.nih.gov/articles/PMC10193984/

This supports the idea that explicit substrate or load-related sensing can legitimately modulate gait, but it does not remove the need to model the correct mechanics if the claim is specifically about crawling physics.

### 3. MuJoCo is a reasonable simulator, but your current usage is narrower than PyElastica-style soft-body scope

Riddle et al. 2025 used MuJoCo for a soft-bodied peristaltic worm robot and emphasized friction and stiffness studies, not a proof that MuJoCo and rod models are interchangeable for every locomotion question.

Source:
- https://pubmed.ncbi.nlm.nih.gov/40934945/

This supports a careful statement:

**The issue is not that MuJoCo cannot model friction. The issue is that this repository currently uses MuJoCo in a simplified rigid-link, scalar-medium regime that does not yet instantiate the anisotropic body-environment mechanics relevant to worm crawling.**

### 4. MuJoCo contact friction is not automatically body-axis anisotropy

MuJoCo’s geom `friction` parameter uses sliding, torsional, and rolling coefficients. The sliding term acts along both tangent-plane axes.

Source:
- https://mujoco.readthedocs.io/en/3.1.5/XMLreference.html

Implication:

Changing `geom_friction = [a, b, c]` is not the same as giving each worm segment different forward vs lateral friction relative to its instantaneous orientation.

## Candidate micro-publication insight

### Best version

**Direct substrate cues are sufficient for a compact biologically inspired controller to switch locomotor dynamics, but realistic crawl-specific gait differentiation in a rigid-link MuJoCo worm likely requires anisotropic body-environment mechanics beyond scalar viscosity and privileged medium labels.**

That gives you:

- a positive finding,
- a clear limitation,
- a concrete next experiment,
- a principled comparison to PyElastica without claiming MuJoCo is inferior.

## Framing for the paper

### Main claim

We show that a minimal NCAP controller with direct environment sensing can robustly modulate oscillator frequency and actuation across water-like and land-like conditions, outperforming memory-heavy alternatives. However, our repository analysis indicates that the current MuJoCo implementation changes locomotion primarily through explicit substrate cues and scalar resistance parameters rather than through true anisotropic contact mechanics. Therefore, mechanically realistic crawling remains an open modeling target rather than a completed result.

### Why this is publishable

This is a clean and defensible systems insight:

- biological direct sensing beats artificial memory for medium switching,
- but controller success should not be overinterpreted as a full mechanics result,
- and anisotropic friction is the critical missing variable for a stronger crawl claim.

That is a useful negative-plus-positive result, especially for a micro-publication.

## Minimal ablation study to make the story stronger

Run three conditions with the same NCAP model and reward setup:

1. `Cue + scalar medium`
   Current setup: explicit environment observation plus viscosity/friction switching.

2. `No cue + scalar medium`
   Remove `environment_type`, `in_water_zone`, `in_land_zone`, and optionally `fluid_viscosity` from observations.

3. `Cue + anisotropy proxy`
   Keep cueing, but add a directional friction proxy based on segment orientation or velocity decomposition relative to the body axis.

Primary outcomes:

- forward speed,
- wavelength/amplitude,
- oscillation frequency,
- transition count,
- body-curvature wave speed,
- water vs land performance gap.

If condition 1 works and condition 2 collapses, your strongest result becomes:

**explicit substrate information, not memory, drives adaptation.**

If condition 3 produces qualitatively different crawling kinematics, then you have the stronger mechanistic result:

**anisotropic mechanics changes the gait class, not just performance.**

## Immediate recommendation

For the micro-publication, do not claim that the current repository already demonstrates anisotropic-friction-based crawling.

Claim instead:

1. a compact biologically inspired controller adapts better with direct environmental sensing than with memory-heavy architectures,
2. the present MuJoCo implementation produces cue-conditioned medium adaptation,
3. true crawl mechanics likely require anisotropic friction or a closer soft-body/contact formulation.

## Suggested collaborator wording

MuJoCo is not the problem by itself. Our current model uses MuJoCo in a simplified rigid-link formulation where locomotion switches are driven mainly by explicit environment cues and scalar viscosity/friction changes. That is enough to test direct sensory modulation of the NCAP controller, but it is not yet a test of anisotropic-friction-driven crawling. PyElastica remains attractive because rod mechanics make local deformation and body-axis-dependent interactions more natural, while our next MuJoCo step should be an ablation on explicit substrate cues and a directional friction proxy.
