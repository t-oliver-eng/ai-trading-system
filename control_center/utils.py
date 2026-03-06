"""
Control Center Utilities
Helper functions for UI, formatting, and common operations
"""

import os
import sys
from datetime import datetime


# ANSI color codes for terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header(text):
    """Print a fancy header"""
    width = 70
    print("\n" + "="*width)
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(width)}{Colors.ENDC}")
    print("="*width)


def print_section(text):
    """Print a section divider"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}")
    print(f"{text}")
    print(f"{'='*70}{Colors.ENDC}")


def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {text}{Colors.ENDC}")


def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}❌ {text}{Colors.ENDC}")


def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.ENDC}")


def print_info(text):
    """Print info message"""
    print(f"{Colors.CYAN}ℹ️  {text}{Colors.ENDC}")


def get_user_choice(prompt, valid_range=None):
    """Get validated user input"""
    while True:
        try:
            choice = input(f"{Colors.BOLD}{prompt}{Colors.ENDC}").strip()
            choice_int = int(choice)
            
            if valid_range and choice_int not in valid_range:
                print_error(f"Please enter a number between {min(valid_range)} and {max(valid_range)}")
                continue
            
            return choice_int
        except ValueError:
            print_error("Please enter a valid number")
        except KeyboardInterrupt:
            print("\n")
            sys.exit(0)


def format_percentage(value, decimals=2):
    """Format value as percentage"""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}%"


def format_currency(value, decimals=2):
    """Format value as currency"""
    if value is None:
        return "N/A"
    return f"${value:,.{decimals}f}"


def format_duration(seconds):
    """Format duration in human readable format"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def print_progress_bar(current, total, prefix='', suffix='', length=50):
    """Print a progress bar"""
    filled = int(length * current // total)
    bar = '█' * filled + '-' * (length - filled)
    percent = 100 * (current / float(total))
    print(f'\r{prefix} |{bar}| {percent:.1f}% {suffix}', end='')
    if current == total:
        print()


def print_table(headers, rows, column_widths=None):
    """Print a formatted table"""
    if not rows:
        print("  (No data)")
        return
    
    # Auto-calculate column widths if not provided
    if column_widths is None:
        column_widths = []
        for i, header in enumerate(headers):
            max_width = len(str(header))
            for row in rows:
                if i < len(row):
                    max_width = max(max_width, len(str(row[i])))
            column_widths.append(max_width + 2)
    
    # Print header
    header_line = "  " + "".join(str(h).ljust(w) for h, w in zip(headers, column_widths))
    print(f"{Colors.BOLD}{header_line}{Colors.ENDC}")
    print("  " + "-" * (sum(column_widths)))
    
    # Print rows
    for row in rows:
        row_line = "  " + "".join(str(cell).ljust(w) for cell, w in zip(row, column_widths))
        print(row_line)


def get_timestamp():
    """Get current timestamp string"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_date_string():
    """Get current date string"""
    return datetime.now().strftime("%Y-%m-%d")


def confirm_action(message):
    """Ask user to confirm an action"""
    response = input(f"\n{Colors.YELLOW}⚠️  {message} (y/n): {Colors.ENDC}").lower()
    return response == 'y'


def wait_for_enter(message="Press Enter to continue..."):
    """Wait for user to press Enter"""
    input(f"\n{message}")
