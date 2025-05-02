#pragma once
enum class Device: __uint8_t { 
    CPU,
    GPU,
    FS,
    AUTO //! should be used for auto select device
};
