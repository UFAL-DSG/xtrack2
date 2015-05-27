import build_data_dstc2

def main():
    build_data_dstc2.main(

    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    main(**vars(args))