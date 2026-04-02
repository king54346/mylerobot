"""
编码工具模块 (Encoding Utils Module)
===================================

提供用于电机通信的数值编码/解码工具函数。

支持的编码格式：
- 符号-幅度表示法 (Sign-Magnitude): 用于某些电机的速度/位置值
- 二进制补码 (Two's Complement): 标准的有符号整数表示

这些工具用于在电机通信协议中转换有符号整数。
"""


def encode_sign_magnitude(value: int, sign_bit_index: int):
    """
    将整数编码为符号-幅度表示法。
    
    符号-幅度表示法使用一个符号位和其余位表示数值的绝对值。
    参考: https://en.wikipedia.org/wiki/Signed_number_representations#Sign%E2%80%93magnitude
    
    Args:
        value: 要编码的整数值
        sign_bit_index: 符号位的位置索引（也是幅度位数）
    
    Returns:
        编码后的无符号整数
    
    Raises:
        ValueError: 如果绝对值超过最大幅度
    """
    max_magnitude = (1 << sign_bit_index) - 1
    magnitude = abs(value)
    if magnitude > max_magnitude:
        raise ValueError(f"Magnitude {magnitude} exceeds {max_magnitude} (max for {sign_bit_index=})")

    direction_bit = 1 if value < 0 else 0
    return (direction_bit << sign_bit_index) | magnitude


def decode_sign_magnitude(encoded_value: int, sign_bit_index: int):
    """
    从符号-幅度表示法解码整数。
    
    参考: https://en.wikipedia.org/wiki/Signed_number_representations#Sign%E2%80%93magnitude
    
    Args:
        encoded_value: 编码后的值
        sign_bit_index: 符号位的位置索引
    
    Returns:
        解码后的有符号整数
    """
    direction_bit = (encoded_value >> sign_bit_index) & 1
    magnitude_mask = (1 << sign_bit_index) - 1
    magnitude = encoded_value & magnitude_mask
    return -magnitude if direction_bit else magnitude


def encode_twos_complement(value: int, n_bytes: int):
    """
    将有符号整数编码为二进制补码表示。
    
    二进制补码是计算机中最常用的有符号整数表示方法。
    参考: https://en.wikipedia.org/wiki/Signed_number_representations#Two%27s_complement
    
    Args:
        value: 要编码的有符号整数
        n_bytes: 字节数（决定位宽和范围）
    
    Returns:
        编码后的无符号整数
    
    Raises:
        ValueError: 如果值超出给定字节数的表示范围
    """

    bit_width = n_bytes * 8
    min_val = -(1 << (bit_width - 1))
    max_val = (1 << (bit_width - 1)) - 1

    if not (min_val <= value <= max_val):
        raise ValueError(
            f"Value {value} out of range for {n_bytes}-byte two's complement: [{min_val}, {max_val}]"
        )

    if value >= 0:
        return value

    return (1 << bit_width) + value


def decode_twos_complement(value: int, n_bytes: int) -> int:
    """
    从二进制补码表示解码整数。
    
    参考: https://en.wikipedia.org/wiki/Signed_number_representations#Two%27s_complement
    
    Args:
        value: 编码后的无符号值
        n_bytes: 字节数
    
    Returns:
        解码后的有符号整数
    """
    bits = n_bytes * 8
    sign_bit = 1 << (bits - 1)
    if value & sign_bit:
        value -= 1 << bits
    return value
