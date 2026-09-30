# Effect of the D2 geometry corrections on test metrics

125 test images, box IoU >= 0.5 + class, conf 0.25, full image, EXIF-upright images. Only the labels of images touched by D2 differ.

| model | labels | recall | precision | F1 | recall duct/terminal/battery |
|---|---|---|---|---|---|
| teacher p5_reg | original | 0.587 | 0.639 | 0.612 | 0.599 |
| teacher p5_reg | D2-corrected | 0.621 | 0.676 | 0.647 | 0.708 |
| student kd_n_p5t_s0 | original | 0.479 | 0.597 | 0.531 | 0.504 |
| student kd_n_p5t_s0 | D2-corrected | 0.501 | 0.625 | 0.556 | 0.577 |
