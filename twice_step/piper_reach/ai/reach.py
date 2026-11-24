"""Run the trained goal-conditioned SAC policy."""

if __package__:
    from .rl.her.evaluate import main
else:
    from rl.her.evaluate import main


if __name__ == "__main__":
    main()


# 本文件主要代码由 AI 助手完成。
