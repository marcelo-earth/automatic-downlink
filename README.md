# automatic-downlink

> A Vision Language Model that runs on-board satellites to automatically decide what's worth downloading to Earth.

**AI in Space Hackathon** (Liquid AI x DPhi Space) | Liquid Track | April-May 2026

## The Problem

Satellites capture terabytes of imagery but can only transmit megabytes per ground station pass. Today, most data is either never analyzed or analyzed days late. Current on-board filtering is limited to narrow CNNs that do one thing (e.g., cloud detection).

## The Solution

A single LFM2.5-VL-450M vision-language model running on-board that:
- Describes every captured image in natural language
- Assigns priority levels (CRITICAL / HIGH / MEDIUM / LOW / SKIP)
- Only downlinks what matters — saving 50%+ bandwidth
- Can be re-tasked via prompt (disaster mode, surveillance mode, etc.) without uploading new weights
