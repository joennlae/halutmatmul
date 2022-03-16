//
//  profile_amm.cpp
//  Bolt
//
//  Created by DB on 12/10/19.
//  Copyright © 2019 D Blalock. All rights reserved.
//

#include <catch2/catch_test_macros.hpp>
#include "profile_amm.hpp"

TEST_CASE("amm mithral", "[amm][matmul][mithral][profile]") {
     std::vector<int> ncodebooks {2, 4, 8, 16, 32, 64};
     std::vector<float> lutconsts {-1, 1, 2, 4};
     _profile_mithral<int8_t>(kCaltechTaskShape0, ncodebooks, lutconsts);
     _profile_mithral<int8_t>(kCaltechTaskShape1, ncodebooks, lutconsts);
     _profile_mithral(kCaltechTaskShape0, ncodebooks, lutconsts);
     _profile_mithral(kCaltechTaskShape1, ncodebooks, lutconsts);
     _profile_mithral(kCifar10TaskShape, ncodebooks, lutconsts);
     _profile_mithral(kCifar100TaskShape, ncodebooks, lutconsts);
     _profile_mithral(kUcrTaskShape0, ncodebooks, lutconsts);
     _profile_mithral(kUcrTaskShape1, ncodebooks, lutconsts);
     _profile_mithral(kUcrTaskShape2, ncodebooks, lutconsts);
}