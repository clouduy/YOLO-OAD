# Contributing to YOLO-OAD 🚀

We welcome and value contributions from the community! YOLO-OAD thrives through collaborative efforts to advance autonomous driving object detection technology. Whether you want to:

- Report a bug or performance issue
- Discuss code improvements and optimizations
- Submit a fix for identified problems
- Propose new features or enhancements
- Help maintain the project

Your contributions help push the boundaries of what's possible in autonomous driving perception systems 😃!

## Submitting a Pull Request (PR) 🛠️

Submitting a PR is straightforward! Here's an example showing how to submit a PR in 4 simple steps:

### 1. Select File to Update

Navigate to the file you want to update (e.g., `requirements.txt`) and click on it in GitHub.

<p align="center"><img width="800" alt="PR_step1" src="https://user-images.githubusercontent.com/26833433/122260847-08be2600-ced4-11eb-828b-8287ace4136c.png"></p>

### 2. Click 'Edit this file'

Locate and click the edit button in the top-right corner of the file view.

<p align="center"><img width="800" alt="PR_step2" src="https://user-images.githubusercontent.com/26833433/122260844-06f46280-ced4-11eb-9eec-b8a24be519ca.png"></p>

### 3. Make Your Changes

Implement your improvements. For example, optimize a module or update documentation.

<p align="center"><img width="800" alt="PR_step3" src="https://user-images.githubusercontent.com/26833433/122260853-0a87e980-ced4-11eb-9fd2-3650fb6e0842.png"></p>

### 4. Preview Changes and Submit PR

Click the **Preview changes** tab to review your modifications. At the bottom, select 'Create a **new branch** for this commit', assign a descriptive branch name like `optimize/dladh-module`, and click the green **Propose changes** button. Your PR is now submitted for review! 😃

<p align="center"><img width="800" alt="PR_step4" src="https://user-images.githubusercontent.com/26833433/122260856-0b208000-ced4-11eb-8e8e-77b6151cbcc3.png"></p>

### PR Best Practices

To ensure smooth integration of your contributions:

- ✅ **Keep PRs Synchronized** – Ensure your PR is up-to-date with the `clouduy/YOLO-OAD` `master` branch. Use the 'Update branch' button or run `git pull` and `git merge master` locally.

<p align="center"><img width="751" alt="Screenshot 2022-08-29 at 22 47 15" src="https://user-images.githubusercontent.com/26833433/187295893-50ed9f44-b2c9-4138-a614-de69bd1753d7.png"></p>

- ✅ **Verify CI Checks** – Ensure all Continuous Integration tests pass before final submission.

<p align="center"><img width="751" alt="Screenshot 2022-08-29 at 22 47 03" src="https://user-images.githubusercontent.com/26833433/187296922-545c5498-f64a-4d8c-8300-5fa764360da6.png"></p>

- ✅ **Focus on Essentials** – Concentrate on the minimal changes needed for your improvement. As Bruce Lee said: *"It is not daily increase but daily decrease, hack away the unessential. The closer to the source, the less wastage there is."*

## Areas of Particular Interest

We especially welcome contributions in these areas aligned with YOLO-OAD's research focus:

- **DLADH Optimizations**: Improvements to the Deformable Lightweight Asymmetric Decoupled Head
- **C3LKSCAA Enhancements**: Better long-range context dependency and receptive field handling
- **C3VGGAM Refinements**: Enhanced local feature extraction and semantic information preservation
- **Focal-EIoU Loss**: Advanced bounding box regression techniques
- **Autonomous Driving Applications**: Real-world performance improvements for traffic objects

## Submitting a Bug Report 🐛

If you encounter issues with YOLO-OAD, please submit a detailed Bug Report!

To help us investigate effectively, we need to reproduce the issue. Follow these guidelines:

When reporting problems, include **code** that others can easily understand and use to **reproduce** the issue. Create a [minimum reproducible example](https://docs.ultralytics.com/help/minimum_reproducible_example/) with these characteristics:

- ✅ **Minimal** – Use the least code needed to reproduce the problem
- ✅ **Complete** – Include all necessary components for reproduction
- ✅ **Reproducible** – Test your code to ensure it reliably demonstrates the issue

Additionally, for effective support:

- ✅ **Current Version** – Ensure you're using the latest code from [master](https://github.com/clouduy/YOLO-OAD/tree/master)
- ✅ **Standard Configuration** – Issues should be reproducible with the original YOLO-OAD implementation ⚠️

If your issue meets these criteria, please submit a new issue using the 🐛 **Bug Report** template and provide a [minimum reproducible example](https://docs.ultralytics.com/help/minimum_reproducible_example/).

## Performance Contributions

Given YOLO-OAD's focus on autonomous driving applications, we particularly value contributions that:

- Improve detection accuracy for traffic objects (vehicles, pedestrians, traffic signs)
- Enhance real-time performance on embedded systems
- Optimize memory usage and computational efficiency
- Extend compatibility with autonomous driving hardware platforms

## License

By contributing to YOLO-OAD, you agree that your contributions will be licensed under the [AGPL-3.0 license](https://choosealicense.com/licenses/agpl-3.0/).

---

*Join us in advancing autonomous driving object detection technology! Your contributions make a real difference in building safer and more reliable self-driving systems.*
