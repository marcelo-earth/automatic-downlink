# Why should satellites run AI?

Hello! Of course, satellites run software to stream images from space to Earth. But why would they run AI software in space, for example? At first, it seems unnecessary. Why not simply download the images to Earth and process them here instead?

## The problem: sending photos home is slow

A satellite circling the Earth takes thousands of photos every day. But it can only connect to a ground station for a few minutes per orbit, and the connection is slow. Imagine taking a thousand photos on your phone but only being able to text five of them before losing signal. So most satellite photos never make it to the ground. They just sit in storage.

Most satellites still handle this the simple way: send everything, in order. A photo of a wildfire waits in the same queue as ten thousand photos of empty ocean. By the time someone on the ground sees the important one, hours have passed. Sometimes days.

Some satellites have started using AI to help with this problem, but each solution is narrow. ESA's PhiSat-1 (launched in 2020) runs a small AI model that detects cloudy images and skips them before sending, which saved about 30% of bandwidth. OroraTech (2025) detects wildfires from thermal images and can send alerts within minutes. Planet's Pelican-4 (2026) can spot specific objects like aircraft at airports.

But each of these AI models does exactly one job. PhiSat-1 can filter clouds, but it can't tell you there's a flood. OroraTech spots fires, but it won't notice deforestation. If you want the satellite to look for something new, you have to train a whole new model and upload it. None of them can explain why an image is important either, they just output a yes/no flag.

## So, what if one model could do all of that?

That's the idea behind automatic-downlink, the project I built. Instead of stacking separate AI models for each task, I put a single "vision language model" (VLM) on the satellite. A VLM is an AI that can look at a photo and describe what's in it using words. You know how chatbots read text and respond? Same idea, but the input is an image.

When the VLM sees flooding in a city, it sends the full image right away. When it sees empty desert, it sends one sentence ("desert terrain, no activity") and moves on. It can also explain its reasoning, something like "urban area with visible flooding, multiple roads submerged, priority: critical."

What I like about this approach is that you can change what it focuses on by rewriting the text instructions (the "prompt"). If you care about fires this week, you say so in the prompt. Ships next week? Just change the text. You don't have to retrain anything or upload new software. As far as I know, no satellite AI system deployed today can do that.

In my tests, the AI cut the data the satellite needed to send by about 95%. Most satellite photos are uninteresting, open water, cloud cover, empty land. Sending a sentence about each of those instead of the full photo frees up the connection for the ones that actually need to reach someone.

## Why does speed matter?

Right now, the gap between a satellite taking a photo and someone on the ground doing something about it is hours. For weather monitoring, that's fine. But for wildfires or floods, hours is the difference between catching something early and cleaning up after. When the AI runs on the satellite, it can flag urgent photos while it's still flying over the area, so the response time goes from hours to minutes.

Satellites aren't the only devices with this problem. Underwater drones, remote sensors in forests, rovers on Mars, they all collect data faster than they can send it home. Building a faster connection is expensive and sometimes physically impossible. Teaching the device to filter what it sends is cheaper. This is sometimes called "edge AI," which just means the AI runs on the device itself, far from any central server.

---

*Built for the Liquid AI x DPhi Space "AI in Space" Hackathon, April 2026. Solo project by Marcelo Arias.*
