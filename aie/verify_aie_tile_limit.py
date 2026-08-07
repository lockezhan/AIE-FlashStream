import json
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Verify AIE tile count limit.')
    parser.add_argument('--limit', type=int, required=True, help='Maximum allowed tile count')
    parser.add_argument('json_file', help='Path to active_cores.json')
    args = parser.parse_args()

    try:
        with open(args.json_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file {args.json_file}: {e}")
        sys.exit(1)

    tile_count = 0
    if "aieIPs" in data and len(data["aieIPs"]) > 0:
        aie_ip = data["aieIPs"][0]
        if "SNoOfCores" in aie_ip:
            tile_count = int(aie_ip["SNoOfCores"])

    if tile_count > args.limit:
        print(f"ERROR: AIE Tile count ({tile_count}) exceeds the limit ({args.limit})!")
        sys.exit(2)
    else:
        print(f"PASS: AIE Tile count ({tile_count}) is within the limit ({args.limit}).")
        sys.exit(0)

if __name__ == '__main__':
    main()
