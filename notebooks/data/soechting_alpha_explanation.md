### Reading alpha (α): what the plus and minus signs mean

Alpha describes the **physical shape**, not its position, height, diameter, or prediction error in centimeters.

- **α = 0:** a true circle.
- **Positive α:** wider left-to-right.
- **Negative α:** longer forward-and-back, toward and away from the body. “Narrow” here does **not** mean vertically elongated in height; all input paths remain in the horizontal plane.
- **Example, α = +0.08:** left–right width is 1.08 times forward depth, or **8% wider than deep**.
- **Example, α = −0.08:** forward depth is 1.08 times left–right width, or **8% deeper than wide**.

Both dimensions change while **area remains constant**. A value of +0.08 does not mean that the circle's original diameter alone was increased by 8% while the other diameter stayed unchanged. This paper's alpha is also not Viviani's signed eccentricity.

**What the numbers in the human/CNN comparison mean:**

- **Human α:** the physical ellipse people judged to be circular, estimated from their wide/narrow responses.
- **CNN α (circularity point):** the physical ellipse needed for the **averaged CNN reconstruction** to have equal left–right and forward–back extents. It is a geometric proxy, not a verbal judgment from the CNN.
- Therefore, a **positive CNN circularity α** means we needed to give it a slightly wide physical ellipse to compensate for its tendency to reconstruct paths as forward-elongated.

**Keep input and output alpha separate:** the `alpha` column identifies the physical input. The `predicted_alpha` column describes a decoded path. A positive input alpha means the input is wide; a positive decoded alpha means the decoded path is wide. A positive *circularity-point* alpha instead identifies the wide input that cancels the model's baseline narrow bias. The signs have the same shape meaning, but the columns refer to different shapes.
