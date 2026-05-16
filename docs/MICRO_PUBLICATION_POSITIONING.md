# Micro-Publication Positioning

Date: 2026-04-01

## What we are sharing

We are sharing a compact, biologically inspired locomotion controller and a controlled simulation framework for mixed-media locomotion.

Concretely, the contribution is:

1. a minimal NCAP-style controller that adapts locomotor dynamics with direct substrate-related input,
2. a curriculum environment that forces transitions between water-like and land-like conditions,
3. an explicit experimental split between cue-driven adaptation and mechanics-driven adaptation.

## What we can claim now

### Strong claim

**Direct environmental sensing enables a small biologically constrained controller to adapt locomotor timing and actuation across media without relying on a large memory architecture.**

This claim is supported by the current codebase and by the existing project narrative comparing biological NCAP against LSTM-based variants.

### Careful claim

**In the current MuJoCo implementation, most observed gait switching should be interpreted as cue-conditioned adaptation under changing scalar resistance, not yet as a full demonstration of anisotropic-friction-driven crawling.**

This is the most important clarity point for the paper.

## What we should not claim yet

We should not claim that:

1. the current repository already reproduces biologically realistic crawling mechanics,
2. MuJoCo has been ruled out as a suitable simulator for worm locomotion,
3. the present crawl/swim distinction emerges purely from contact mechanics alone.

Those statements would overreach the current implementation.

## Why this is useful

This work is useful because it contributes a clean separation of controller and mechanics questions.

Most related work tends to mix together:

1. richer body mechanics,
2. richer sensing,
3. larger controllers,
4. changing reward structure.

Your repo is valuable if it shows that a great deal of adaptive behavior can already come from:

- a very small controller,
- biologically constrained architecture,
- direct environmental sensing,
- careful task design.

That is relevant to computational neuroscience, embodied AI, and bio-inspired robotics because it argues that:

**adaptive locomotion does not necessarily require large generic memory systems.**

## Why the anisotropy exploration matters

The anisotropy experiment is the bridge from a controller paper to a stronger mechanics paper.

If privileged environment cues are removed and the behavior stays adaptive, that suggests the controller can exploit embodied dynamics rather than only explicit labels.

If a directional drag proxy changes the qualitative gait class, that shows that body-environment mechanics are not just quantitative modifiers of speed but qualitative determinants of locomotor strategy.

That would make the work more relevant to:

- worm biomechanics,
- soft-robot design,
- simulator-comparison studies,
- morphology-controller co-design.

## Best paper story

The most defensible story is:

1. Minimal biological controllers outperform memory-heavy ones for medium switching.
2. The current adaptive effect is largely driven by direct substrate cues and scalar resistance changes.
3. Introducing anisotropic mechanics is the next critical step toward realistic crawling.

This gives the paper a useful message even before full anisotropic crawling is solved:

**we can already isolate what part of adaptation comes from sensing and what part still depends on missing mechanics.**

## Recommended paper title direction

Possible title directions:

1. Direct environmental sensing supports adaptive locomotion in a minimal biologically inspired swimmer-crawler controller
2. Separating substrate cueing from mechanics in a biologically inspired controller for mixed-media locomotion
3. Minimal neural circuit priors support medium adaptation, while anisotropic mechanics remain critical for realistic crawling

## Recommended experiment matrix

Use three conditions:

1. `baseline`
   Explicit substrate cues on, anisotropic drag off.

2. `cue_ablation`
   Explicit substrate cues off, anisotropic drag off.

3. `anisotropy_proxy`
   Explicit substrate cues on or off, anisotropic drag proxy on.

If only one new result is needed for the micro-publication, the highest-value comparison is:

**baseline vs cue_ablation**

If a second result is feasible, add:

**cue_ablation vs anisotropy_proxy**

That will answer:

- how much of the current effect is privileged sensing,
- whether directional mechanics changes the qualitative behavior,
- what exactly the paper is contributing.

## New CLI experiments

The repository now exposes simple switches for these conditions:

Baseline:

```bash
python main.py --mode train_curriculum --model_type enhanced_ncap
```

Cue ablation:

```bash
python main.py --mode train_curriculum --model_type enhanced_ncap --hide_environment_observation --hide_viscosity_observation
```

Anisotropy proxy:

```bash
python main.py --mode train_curriculum --model_type enhanced_ncap --anisotropic_drag_mode proxy --anisotropic_drag_ratio 10 --anisotropic_drag_gain 0.02
```

## Practical summary

What we are sharing is not "we solved worm crawling."

What we are sharing is:

**a clear demonstration that biologically constrained, low-parameter control can adapt across media, plus an explicit diagnosis of the mechanics still missing for a stronger crawling claim.**

That is useful because it sharpens the scientific question instead of obscuring it.
